# coding=utf-8
# Copyright 2025 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

logger = logging.get_logger(__name__)


class MoonViTConfig(PretrainedConfig):
    model_type = "moonvit"

    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 27,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        merge_kernel_size: tuple[int, int] = (2, 2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        # Positional embedding config
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        # Transformer config
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # Patch merger config
        self.merge_kernel_size = merge_kernel_size


class KimiVLConfig(PretrainedConfig):
    model_type = "kimi_vl"

    def __init__(
        self,
        vision_config: Optional[Union[dict, MoonViTConfig]] = None,
        text_config: Optional[Union[dict, DeepseekV3Config]] = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        **kwargs,
    ):
        if vision_config is None:
            vision_config = MoonViTConfig()
        elif isinstance(vision_config, dict):
            vision_config = MoonViTConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = DeepseekV3Config()
        elif isinstance(text_config, dict):
            text_config = DeepseekV3Config(**text_config)
        self.text_config = text_config

        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id

        attn_implementation = kwargs.get("attn_implementation")
        if attn_implementation is not None:
            if attn_implementation in ["eager", "flash_attention_2"]:
                self._attn_implementation = attn_implementation
                self.vision_config._attn_implementation = attn_implementation
                self.text_config._attn_implementation = attn_implementation
            else:
                raise ValueError(
                    f"Invalid attention implementation: {attn_implementation}"
                )

        super().__init__(pad_token_id=pad_token_id, **kwargs)


__all__ = ["MoonViTConfig", "KimiVLConfig"]

