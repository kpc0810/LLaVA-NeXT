import re
import copy
import math
import random
import pickle
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM, LlavaQwenModel, LlavaQwenConfig
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM, unpad_image
from llava.model.utils import parse_object_nouns_and_action_verbs, get_phrase_indices
from transformers.generation.utils import GenerateOutput
from llava.model.mixin import HalluGenerationMixin

from llava.model.contrastive_projector.builder import build_contrastive_projector
from llava.model.act_squeezer.builder import build_act_squeezer
from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print, rank0_breakpoint

from deepspeed import comm as dist

torch.autograd.set_detect_anomaly(True)

def is_dist_avail_and_initialized():
    """Refer to LAVIS, lavis.common.dist_utils.is_dist_avail_and_initialized"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def concat_all_gather(tensor):
    """Refer to LAVIS, lavis.models.base_model.concat_all_gather
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = torch.distributed.get_world_size()  # Get the total number of processes in the distributed environment.
    if world_size == 1:  # If there is only one process, no need to gather data from other ranks.
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)  # Serialize the input data into a byte stream using pickle.
    storage = torch.ByteStorage.from_buffer(buffer)  # Convert the byte stream into a PyTorch ByteStorage.
    tensor = torch.ByteTensor(storage).to("cuda")  # Convert the storage into a ByteTensor and move it to GPU memory.

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")  # Get the size (number of elements) of the local tensor.
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]  # Prepare a list to store tensor sizes from all ranks.
    dist.all_gather(size_list, local_size)  # Gather the sizes of tensors from all ranks into size_list.
    size_list = [int(size.item()) for size in size_list]  # Convert the sizes to a list of integers.
    max_size = max(size_list)  # Determine the maximum size among all tensors.

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []  # Initialize a list to store tensors from all ranks.
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))  # Create a padded ByteTensor for each rank.
    if local_size != max_size:  # If the local tensor size is smaller than the max size, pad it.
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")  # Create padding bytes.
        tensor = torch.cat((tensor, padding), dim=0)  # Concatenate the padding to the local tensor.
    dist.all_gather(tensor_list, tensor)  # Gather the padded tensors from all ranks.

    data_list = []  # Initialize a list to store the deserialized data from all ranks.
    for size, tensor in zip(size_list, tensor_list):  # Iterate over each rank's tensor and its size.
        buffer = tensor.cpu().numpy().tobytes()[:size]  # Extract the valid part of the tensor and convert it back to bytes.
        data_list.append(pickle.loads(buffer))  # Deserialize the byte stream back into the original object.

    return data_list  # Return the list of gathered data from all ranks.


@dataclass
class ContrastiveCausalLMOutput(CausalLMOutputWithPast):
    cl_temp_log: Optional[float] = None
    cl_loss_log: Optional[dict] = None


class FaithLlavaQwenConfig(LlavaQwenConfig):
    model_type = "faith_llava_qwen"


class FaithLlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = FaithLlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(FaithLlavaQwenModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            self.contrastive_vision_projector = build_contrastive_projector(config, modality='vision')
            self.contrastive_text_projector = build_contrastive_projector(config, modality='text')
            self.itc_temp = nn.Parameter(torch.ones([]) * 0.07)
            self.act_squeezer = build_act_squeezer(config)
        
    def initialize_vision_modules(self, model_args, fsdp=None):
        super(FaithLlavaQwenModel, self).initialize_vision_modules(model_args, fsdp)
        self.initialize_contrastive_projector(model_args)
        self.initialize_act_squeezer(model_args)
    
    def initialize_contrastive_projector(self, config):
        if getattr(self, "contrastive_vision_projector", None) is None:
            self.contrastive_vision_projector = build_contrastive_projector(config, modality='vision')
        if getattr(self, "contrastive_text_projector", None) is None:
            self.contrastive_text_projector = build_contrastive_projector(config, modality='text')
        if getattr(self, "itc_temp", None) is None:
            self.itc_temp = nn.Parameter(torch.ones([]) * 0.07)
    
    def initialize_act_squeezer(self, config):
        if getattr(self, "act_squeezer", None) is None:
            self.act_squeezer = build_act_squeezer(config)
        

class FaithLlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM, HalluGenerationMixin):
    config_class = FaithLlavaQwenConfig
    
    def __init__(self, config):
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None
        
        self.model = FaithLlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        
    def get_model(self):
        return self.model
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.caption_prefix_ids = torch.tensor(self.tokenizer("<|im_end|>\n<|im_start|>assistant\n").input_ids, dtype=torch.long)

    def return_glob_avg_pooling_hidden(self, last_hidden_state, split_indices):
        """
        Args:
            last_hidden_state: [bs, seq_len, hidden_dim]
            split_indices: [bs, num_spans, 2]
                ** (1) visual_global_indices: [bs, # of frames = 8, 2]
                   (2) lang_global_indices: [bs, # of captions = 1, 2]
        Returns:
            pool_embeds: [bs, hidden_dim]
        """
        batch_size = last_hidden_state.shape[0]  # 16
        num_spans = split_indices.shape[1]
        device = last_hidden_state.device
        dtype = last_hidden_state.dtype
        
        split_input_embeds = []  # list of list of tensor, shape: [bs, num_spans, span_len, hidden_dim], note that `span_len` is not fixed as split_indices is caption split.
        for last_hidden_state_per_sample, split_indices_per_sample in zip(last_hidden_state, split_indices):
            split_input_embeds_per_sample = []
            num_spans = split_indices_per_sample.shape[0]
            for i in range(num_spans):
                start_idx, end_idx = split_indices_per_sample[i, 0], split_indices_per_sample[i, 1]
                span_len = end_idx - start_idx + 1
                split_input_embeds_per_sample.append(last_hidden_state_per_sample[start_idx:end_idx+1, :])
            split_input_embeds.append(split_input_embeds_per_sample)
        
        pool_embeds = torch.tensor([]).to(device).to(dtype) # if coarse, expected shape: [bs, num_spans, hidden_dim]
        for split_input_embeds_per_sample in split_input_embeds:
            pool_embeds_per_sample = torch.tensor([]).to(device).to(dtype)  # expected shape: [num_spans, hidden_dim]
            for span_input_embeds in split_input_embeds_per_sample:
                span_input_embeds = span_input_embeds.mean(dim=0).unsqueeze(dim=0)
                pool_embeds_per_sample = torch.cat((pool_embeds_per_sample, span_input_embeds), dim=0)
            pool_embeds_per_sample = pool_embeds_per_sample.unsqueeze(dim=0)
            pool_embeds = torch.cat((pool_embeds, pool_embeds_per_sample), dim=0)
        
        if pool_embeds.shape[1] == 1:
            pool_embeds = pool_embeds.squeeze(dim=1)
        else:
            pool_embeds = pool_embeds.mean(dim=1)
        
        return pool_embeds
    
    def compute_vtcc_loss(self, last_hidden_state, hallu_last_hidden_state, new_frame_bounds, new_caption_bounds, hallu_caption_bounds):
        """
        Args:
            last_hidden_state (torch.FloatTensor): [bs, seq_len, hidden_dim]
            hallu_last_hidden_state (torch.FloatTensor): [bs, seq_len, hidden_dim]
            new_frame_bounds (torch.LongTensor): [bs, num_frames = 8, 2]
            new_caption_bounds (torch.LongTensor): [bs, num_captions = 1, 2]
            hallu_caption_bounds (torch.LongTensor): [bs, num_hallu_captions = 1, 2]

        Returns:
            loss_vtcc (torch.FloatTensor): [bs, 1]
        """
        
        with torch.no_grad():
            self.model.itc_temp.data.clamp_(0.01, 0.5)
        
        glob_video_embedding = self.return_glob_avg_pooling_hidden(last_hidden_state=last_hidden_state, split_indices=new_frame_bounds)
        glob_video_embedding = self.model.contrastive_vision_projector(glob_video_embedding)
        all_video_feat = concat_all_gather(glob_video_embedding)  # [bs * num_process, hidden_dim]
        
        glob_text_embedding = self.return_glob_avg_pooling_hidden(last_hidden_state=last_hidden_state, split_indices=new_caption_bounds)
        glob_text_embedding = self.model.contrastive_text_projector(glob_text_embedding)
        all_text_feat = concat_all_gather(glob_text_embedding)  # [bs * num_process, hidden_dim]
        
        if hallu_last_hidden_state is not None:
            glob_neg_text_embedding = self.return_glob_avg_pooling_hidden(last_hidden_state=hallu_last_hidden_state, split_indices=hallu_caption_bounds)
            glob_neg_text_embedding = self.model.contrastive_text_projector(glob_neg_text_embedding)
            all_neg_text_feat = concat_all_gather(glob_neg_text_embedding)  # [bs * num_process, hidden_dim]
            all_text_feat = torch.cat([all_text_feat, all_neg_text_feat], dim=0)  # [bs * num_process * 2, hidden_dim]
        
        sim_v2t = F.normalize(glob_video_embedding,dim=-1) @ F.normalize(all_text_feat,dim=-1).T / self.model.itc_temp  # [bs, bs * num_process (*2)]
        sim_t2v = F.normalize(glob_text_embedding,dim=-1) @ F.normalize(all_video_feat,dim=-1).T / self.model.itc_temp  # [bs, bs * num_process]

        rank, bs, device = dist.get_rank(), last_hidden_state.size(0), last_hidden_state.device
        sim_labels = torch.arange(rank * bs, rank * bs + bs, dtype=torch.long).to(device)
        
        loss_v2t_vtcc = F.cross_entropy(sim_v2t, sim_labels, label_smoothing=0.1)
        loss_t2v_vtcc = F.cross_entropy(sim_t2v, sim_labels, label_smoothing=0.1)
        loss_vtcc = (loss_v2t_vtcc + loss_t2v_vtcc) / 2
        return loss_vtcc
    
    def get_aggr_embedding(self, embedding, indices):
        """
        Args:
            embedding (torch.FloatTensor): [seq_len, hidden_dim]
            indices (List[List[torch.LongTensor]]): [num_objects, occupied_indices]
        Returns:
            aggr_embedding (torch.FloatTensor): [num_objects, hidden_dim]
        """        
        aggr_embedding = torch.tensor([]).to(embedding.device).to(embedding.dtype)
        for obj_idx, occupied_indices in enumerate(indices):
            aggr_embedding = torch.cat((aggr_embedding, embedding[occupied_indices, :].mean(dim=0).unsqueeze(dim=0)), dim=0)
        return aggr_embedding
    
    def compute_cl_loss(
        self,
        last_hidden_state: torch.FloatTensor,
        hallu_last_hidden_state: torch.FloatTensor,
        # video-caption contrastive learning
        new_frame_bounds: torch.LongTensor,
        new_caption_bounds: torch.LongTensor,
        hallu_caption_bounds: torch.LongTensor
    ):
        """
        Args:
            last_hidden_state (torch.FloatTensor): [bs, seq_len, hidden_dim]
            hallu_last_hidden_state (torch.FloatTensor): [bs, hallu_seq_len, hidden_dim]
            
            new_frame_bounds (torch.LongTensor): [bs, num_frames = 8, 2]
            new_caption_bounds (torch.LongTensor): [bs, num_caption = 1, 2]
            hallu_caption_bounds (torch.LongTensor): [bs, num_hallu_captions = 1, 2]
        Returns:
            cl_loss (dict): the contrastive learning loss
        """
        cl_loss = {}
        cl_loss['loss_vtcc'] = self.compute_vtcc_loss(
            last_hidden_state=last_hidden_state, 
            hallu_last_hidden_state=hallu_last_hidden_state, 
            new_frame_bounds=new_frame_bounds, 
            new_caption_bounds=new_caption_bounds, 
            hallu_caption_bounds=hallu_caption_bounds
        )
        return cl_loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        # evolving hallucinated negative
        hallu_input_ids: Optional[torch.LongTensor] = None,
        hallu_attention_mask: Optional[torch.Tensor] = None,
        # others
        hallu_forward: Optional[bool] = False,
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids, 
                position_ids, 
                attention_mask, 
                past_key_values, 
                inputs_embeds, 
                labels, 
                frame_bounds, 
                caption_bounds
            ) = self.prepare_contrastive_inputs_labels_for_multimodal(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=labels,
                images=images,
                modalities=modalities,
                image_sizes=image_sizes,
            )

        # generate hallucinated hard negative on-the-fly
        # === caption loss ===
        output_hidden_states=True
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not self.training or hallu_forward:
            return outputs

        # === contrastive learning ===
        if self.config.tune_contrastive_projector:

            loss = outputs.loss

            caption_loss = loss.item()
            last_hidden_state = outputs.hidden_states[-1]
            
            if self.config.use_hard_neg:
                hallu_pseudo_labels = hallu_input_ids.masked_fill(hallu_input_ids == self.tokenizer.pad_token_id, IGNORE_INDEX)
                hallu_attention_mask = hallu_input_ids.ne(self.tokenizer.pad_token_id)
                (
                    _,
                    _,
                    hallu_attention_mask,
                    _,
                    hallu_inputs_embeds,
                    hallu_pseudo_labels,  # dummy, will not be used
                    _,
                    hallu_caption_bounds
                ) = self.prepare_contrastive_inputs_labels_for_multimodal(
                    input_ids=hallu_input_ids, 
                    position_ids=None, 
                    attention_mask=hallu_attention_mask, 
                    past_key_values=None, 
                    labels=hallu_pseudo_labels, 
                    images=images,
                    modalities=modalities, 
                    image_sizes=image_sizes
                )
                hard_neg_inputs_embeds = hallu_inputs_embeds
                with torch.inference_mode():
                    hallu_last_hidden_state = super().forward(
                        attention_mask=hallu_attention_mask,
                        inputs_embeds=hard_neg_inputs_embeds,
                        output_hidden_states=True,
                    ).hidden_states[-1].detach()
            else:
                hallu_last_hidden_state = None
                hallu_caption_bounds = None
                
            self.model.contrastive_vision_projector.train()
            self.model.contrastive_text_projector.train()

            cl_loss = self.compute_cl_loss(
                last_hidden_state=last_hidden_state, 
                hallu_last_hidden_state=hallu_last_hidden_state, 
                new_frame_bounds=frame_bounds, 
                new_caption_bounds=caption_bounds, 
                hallu_caption_bounds=hallu_caption_bounds
            )

            if len(cl_loss) != 0:
                cl_loss_log = {}
                for loss_name, loss_value in cl_loss.items():
                    if loss_value is not None:
                        cl_loss_log[loss_name] = loss_value.detach().clone()

                for loss_name, loss_value in cl_loss.items():
                    if loss_name == 'loss_vtcc':
                        loss = loss + loss_value * self.config.vccl_wt

                # create ContrastiveCausalLMOutput object
                outputs = ContrastiveCausalLMOutput(**outputs.__dict__)
                outputs.cl_temp_log = self.model.itc_temp.item()
                outputs.cl_loss_log = {loss_name: loss_value.item() for loss_name, loss_value in cl_loss_log.items()}
                # report
                self.report_metrics(cl_temperature=outputs.cl_temp_log, caption_loss=caption_loss, **outputs.cl_loss_log)
                rank_print(f"cl_loss_log = {outputs.cl_loss_log}; caption_loss = {caption_loss}")
            outputs.loss = loss

        return outputs
    
    def prepare_contrastive_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes):
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)  # shape = (bs * num_frames, 3, 384, 384)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = self.encode_images(concat_images)  # shape = (bs * num_frames, patch * patch = 27 * 27, dim), patch is the number of patches per side after visual encoding by conv2d in VisualEmbedding
            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)


            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)  # list of tensors: [bs, (num_frames, patch * patch, dim)]
            image_features = []  # list of tensors: [bs, (num_frames, h * w, dim)], h * w are pooled results of the patch * patch
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):  # shape of image_feature = (frame_num, num_patches = h * w, dim)
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature)  # Here, image_feature shape = (num_frames * h * (w+1), dim)
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)

                                # print("!!!!!!!!!!!!")
                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # init the return variables
        new_input_embeds, new_labels, new_frame_bounds, new_caption_bounds = [], [], [], []
        cur_image_idx = 0
        # import pdb; pdb.set_trace()
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]  # note that image_features (List[torch.FloatTensor]) = (bs, (num_frames * h * (w+1))) 
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            
            ## Create (1) cur_input_ids_noim, (2) cur_labels_noim, (3) cur_input_embeds_noim (no image tokens),
            ## which are in the format of list of tensors (split by image tokens)
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_noim = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            ## `cur_new_input_embeds` and `cur_new_labels` are list of tensors (split by image tokens)
            ## besides, they are interleaved with text and image features
            cur_new_input_embeds = []
            cur_new_labels = []
            dtype, device = cur_input_embeds.dtype, cur_input_embeds.device
            caption_prefix_ids = self.caption_prefix_ids.to(device)
            cur_new_input_embeds, cur_new_labels = torch.tensor([]).to(dtype).to(device), torch.tensor([]).to(torch.long).to(device)
            cur_frame_bounds, cur_caption_bound = torch.tensor([]).to(device), torch.tensor([]).to(device)
            for i in range(num_images + 1):
                cur_new_input_embeds = torch.cat((cur_new_input_embeds, cur_input_embeds_noim[i]))
                cur_new_labels = torch.cat((cur_new_labels, cur_labels_noim[i]))
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds = torch.cat((cur_new_input_embeds, cur_image_features))
                    cur_new_labels = torch.cat((cur_new_labels, torch.full(
                        (cur_image_features.shape[0],),
                        IGNORE_INDEX,
                        device=cur_labels.device,
                        dtype=cur_labels.dtype
                    )))
                    # new_frame_bounds
                    frame_end_idx = cur_new_input_embeds.shape[0]
                    frame_start_idx = frame_end_idx - cur_image_features.shape[0]
                    cur_frame_bounds = torch.cat((cur_frame_bounds, torch.tensor([frame_start_idx, frame_end_idx]).unsqueeze(0).to(device)))
                if i == num_images:
                    fold_cur_new_labels = cur_new_labels.unfold(0, len(caption_prefix_ids), 1)
                    prefix_cap_start_idx = (fold_cur_new_labels == caption_prefix_ids.expand([fold_cur_new_labels.shape[0], len(caption_prefix_ids)])).all(dim=-1).nonzero(as_tuple=True)[-1]
                    # new_caption_bounds
                    cap_start_idx = prefix_cap_start_idx + len(caption_prefix_ids)
                    cap_end_idx = cur_new_input_embeds.shape[0]
                    cur_caption_bound = torch.tensor([cap_start_idx, cap_end_idx]).unsqueeze(0).to(device)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_frame_bounds.append(cur_frame_bounds)
            new_caption_bounds.append(cur_caption_bound)
        
        frame_num = images[0].shape[0]
        new_frame_bounds = torch.stack(new_frame_bounds).to(torch.int32)
        new_frame_bounds = self._split_frame_bounds(new_frame_bounds, frame_num)
        new_caption_bounds = torch.stack(new_caption_bounds).to(torch.int32)
        
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        if tokenizer_model_max_length is not None:
            if any(len(x) > tokenizer_model_max_length for x in new_input_embeds):
                warnings.warn("Inputs truncated!")
            
            new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
            new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
            
            # fix the len for each sample
            max_len_idx = tokenizer_model_max_length - 1
            new_frame_bounds = torch.where(new_frame_bounds < max_len_idx, new_frame_bounds, max_len_idx)
            new_caption_bounds = torch.where(new_caption_bounds < max_len_idx, new_caption_bounds, max_len_idx)

            
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, \
            new_frame_bounds, new_caption_bounds

    def _split_frame_bounds(self, frame_bounds, frame_num):
        """ Since the image tokens are concated together in LLaVA-Video, we need to split the frame bounds into frame_num chunks accordingly.
        Args:
            frame_bounds (torch.Tensor): The frame bounds to be split. Shape = (bs, 1, 2).
            frame_num (int): The number of frames in the video.
        Returns:
            chunked_frame_bounds (torch.Tensor): Frame bounds. Shape = (bs, frame_num, 2).
        """
        # chunk the frame bounds
        chunked_frame_bounds = []
        for batch_idx in range(frame_bounds.shape[0]):
            start_vdo_idx, end_vdo_idx = frame_bounds[batch_idx].squeeze()
            assert (end_vdo_idx - start_vdo_idx) % frame_num == 0, "The number of frames must be divisible by the number of chunks"
            chunk_size = (end_vdo_idx - start_vdo_idx) // frame_num
            chunked_frame_bounds.append(torch.stack([torch.tensor([start_vdo_idx + i * chunk_size, start_vdo_idx + (i + 1) * chunk_size]) for i in range(frame_num)], dim=0))
        chunked_frame_bounds = torch.stack(chunked_frame_bounds)
                    
        return chunked_frame_bounds

    # directly copy from LlavaQwenForCausalLM
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    # directly copy from LlavaQwenForCausalLM
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs

AutoConfig.register("faith_llava_qwen", FaithLlavaQwenConfig)
AutoModelForCausalLM.register(FaithLlavaQwenConfig, FaithLlavaQwenForCausalLM)