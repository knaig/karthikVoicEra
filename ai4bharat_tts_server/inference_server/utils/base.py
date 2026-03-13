"""Base class and weight init for Parler-TTS decoder stack."""

import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

from ..configuration_parler_tts import ParlerTTSDecoderConfig


class ParlerTTSPreTrainedModel(PreTrainedModel):
    """Base for Parler-TTS decoder models; handles config and weight initialization."""

    config_class = ParlerTTSDecoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _no_split_modules = ["ParlerTTSDecoderLayer", "ParlerTTSAttention"]
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_factor
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
