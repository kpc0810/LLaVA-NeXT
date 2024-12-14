import os
import torch

from .base_projector import ContrastiveProjectorConfig, ContrastiveProjector
from transformers import PretrainedConfig, PreTrainedModel

def build_contrastive_projector(
    config: PretrainedConfig, modality: str
) -> PreTrainedModel:
    cp_config = ContrastiveProjectorConfig(config)
    return ContrastiveProjector(cp_config)
