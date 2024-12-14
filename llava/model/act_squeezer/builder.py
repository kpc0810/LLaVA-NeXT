import os
import torch

from .base_projector import ActSqueezerConfig, ActSqueezer
from transformers import PretrainedConfig, PreTrainedModel

def build_act_squeezer(config: PretrainedConfig) -> PreTrainedModel:
    as_config = ActSqueezerConfig(config)
    return ActSqueezer(as_config)