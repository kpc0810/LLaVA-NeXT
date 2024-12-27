import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math
import warnings

from torch import einsum
from einops import rearrange, repeat, reduce
from einops_exts import rearrange_many, repeat_many
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from llava.utils import rank0_breakpoint


class ActSqueezerConfig(PretrainedConfig):
    model_type = "act_squeezer"

    def __init__(self, config: PretrainedConfig = None, **kwargs):
        super().__init__(**kwargs)
        if config is not None:
            self.query_dim = config.hidden_size  # llm's output hidden size (pre-trained: 768)
            self.kv_as_dim = config.hidden_size  # llm's output hidden size (pre-trained: 1024)
            self.num_latents = 16
            self.dim_as_head = 96
            self.as_heads = 8
            self.as_depth = 1
            self.as_ff_mult = 2
            self.model_dtype = config.torch_dtype
            self.encoder_max_len = 1792
            self.pretrained_model_path = "checkpoints/pretrained/act_squeezer/actionbench_ssv2_patch_and_fuse.pth"


def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )


class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        k_v_dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(k_v_dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(k_v_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, latents, attn_mask=None, attn_guidance=None):
        """
        einstein notation
            b - batch
            t - time
            n - sequence
            d - dimension
        x: vision features (batch_size(b), num_video(t), num_tokens(n), k_v_dim(d))
        attn_mask: (batch_size, num_video, num_tokens)
        attn_guidance: a weighting for the x (i.e., raw_vision_features): (batch_size, num_video, num_tokens)
        latents: learnable query (b, t, num_latents, dim)
        """

        x = self.norm_media(x)
        latents = self.norm_latents(latents)
        
        b, t, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # get key value based on x
        # kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(x).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)
        q = q * self.scale
        # q_head_dim = q.shape[-1]

        # attention
        sim = einsum('... i d, ... j d  -> ... i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        # sim = sim / math.sqrt(q_head_dim)  # Follow huggingface perceiver

        if attn_mask is not None:
            attn_mask = rearrange(attn_mask, "b t n -> b 1 t 1 n")
            sim = sim.masked_fill(attn_mask.bool(), torch.finfo(x.dtype).min)

        if attn_guidance is not None:
            attn_guidance = repeat(attn_guidance, "b t j -> b h t i j", h=h, i=sim.shape[-2])
            sim = sim * attn_guidance

        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        return self.to_out(out)


class Perceiver(nn.Module):
    def __init__(self, config: ActSqueezerConfig):
        super().__init__()
        self.config = config
        self.latents = nn.Parameter(torch.randn(config.num_latents, config.query_dim))
        self.layers = nn.ModuleList([])
        for i in range(config.as_depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(dim=config.query_dim, 
                                       k_v_dim=config.kv_as_dim, 
                                       dim_head=config.dim_as_head, 
                                       heads=config.as_heads),
                    FeedForward(dim=config.query_dim, mult=config.as_ff_mult),
                ])
            )
            
        self.norm = nn.LayerNorm(config.query_dim)
        
    def forward(self, x, latents=None, attn_mask=None, attn_guidance=None):
        # x: vision feature that provides key and value
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')
        if attn_guidance is not None and attn_guidance.ndim == 2:
            attn_guidance = rearrange(attn_guidance, 'b n -> b 1 n')

        ## assume that if input has multiple frames, pos_embed already included
        # times = x.shape[1]
        # x = x + self.media_pos_emb[:times] # expected shape: (batch_size, num_video, num_tokens (patches*frms), dimension)

        if latents is None:
            # use unconditional latents
            latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])
        else:
            # use inputed conditional latents
            latents = repeat(latents, 'b 1 n d -> b m n d', b = x.shape[0], m = x.shape[1])

        for attn, ff in self.layers:
            latents = attn(x, latents, attn_mask=attn_mask, attn_guidance=attn_guidance) + latents
            latents = ff(latents) + latents

        return self.norm(latents)
        

class ActSqueezer(PreTrainedModel):
    config_class = ActSqueezerConfig
    
    def __init__(self, config: ActSqueezerConfig):
        super().__init__(config)
        
        self.config = config
        self.vision_perceiver = Perceiver(config)
        self.pretrained_model_path = config.pretrained_model_path
        

        self.__init_pretrained_weights(self.pretrained_model_path)

    def __weight_tile_and_truncate(self, input_tensor, target_shape):
        """
        Tiles and truncates an input tensor to match a target shape by repeating and slicing.
        
        Args:
            input_tensor (torch.Tensor): Input tensor to be resized
            target_shape (tuple): Desired output shape, must be 1D or 2D
            
        Returns:
            torch.Tensor: Resized tensor matching target_shape
            
        Raises:
            ValueError: If target_shape has more than 2 dimensions
            AssertionError: If input and target shapes have different number of dimensions
        """
        assert len(input_tensor.shape) == len(target_shape), f"input_tensor shape: {input_tensor.shape}; target_shape: {target_shape}"
        
        if len(target_shape) == 1:
            tile_w = math.ceil(target_shape[0] / input_tensor.shape[0])
            input_tensor = input_tensor.repeat(tile_w)
            input_tensor = input_tensor[:target_shape[0]]
        elif len(target_shape) == 2:
            tile_h, tile_w = math.ceil(target_shape[0] / input_tensor.shape[0]), math.ceil(target_shape[1] / input_tensor.shape[1])
            input_tensor = input_tensor.repeat(tile_h, tile_w)
            input_tensor = input_tensor[:target_shape[0], :target_shape[1]]
        else:
            raise ValueError(f"target_shape: {target_shape} is not supported")
        
        return input_tensor

    def __init_pretrained_weights(self, pretrained_model_name_or_path):
        pretrained_weights = torch.load(pretrained_model_name_or_path)
        name_to_weights = {}
        for model_name in pretrained_weights.get("model"):
            if "vision_perceiver" in model_name:
                name_to_weights[model_name] = pretrained_weights.get("model")[model_name]
                # find the corresponding layer in self
                self_correspond_layer = self
                for layer_name in model_name.split("."):
                    if layer_name.isdigit():
                        self_correspond_layer = self_correspond_layer[int(layer_name)]
                    else:
                        self_correspond_layer = getattr(self_correspond_layer, layer_name)

                filled_pretrained_weights = self.__weight_tile_and_truncate(name_to_weights[model_name], self_correspond_layer.shape)
                name_to_weights[model_name] = filled_pretrained_weights

        self.load_state_dict(name_to_weights)
        
    def forward(self, x: torch.LongTensor, attn_mask: torch.LongTensor, attn_guidance: torch.LongTensor = None):
        """_summary_

        Args:
            x (torch.LongTensor), shape = [bs, all_num_patches, dim]: the object visual embeddings related to the specific action
            attn_mask (torch.LongTensor), shape = [bs, all_num_patches]: the attention mask for the object visual embeddings
            attn_guidance (torch.LongTensor), shape = [bs, all_num_patches]: the attention guidance for the object visual embeddings
        Returns:
            torch.FloatTensor: shape = [bs, num_latents, dim]
        """
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # [bs, 1, all_num_patches]
        if attn_guidance is not None:
            attn_guidance = attn_guidance.unsqueeze(1)  # [bs, 1, all_num_patches]
        out = self.vision_perceiver(x, latents=None, attn_mask=attn_mask, attn_guidance=attn_guidance)[:,0,:,:]  # [bs, num_latents, dim_latents]
        return out


AutoConfig.register("act_squeezer", ActSqueezerConfig)
AutoModel.register(ActSqueezerConfig, ActSqueezer)

if __name__ == "__main__":
    config = ActSqueezerConfig()
    model = ActSqueezer(config)
    print(model)