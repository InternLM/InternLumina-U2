# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
from abc import ABC
from typing import Callable, Dict, List, Optional

import torch
from transformers import PretrainedConfig, PreTrainedModel


class BaseFoundationConfigMixin(PretrainedConfig, ABC):
    def __init__(self, vocab_size: int = 0, hidden_size: int = 0, tie_word_embeddings: bool = None, **kwargs):
        # A Foundation model must contain `vocab_size`, `hidden_size`, and `tie_word_embeddings`
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(
            vocab_size=vocab_size, hidden_size=hidden_size, tie_word_embeddings=tie_word_embeddings, **kwargs
        )


def base_position_id_func(input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
    if len(input_ids.shape) == 1:
        input_ids = input_ids.unsqueeze(0)
    if len(attention_mask.shape) == 1:
        attention_mask = attention_mask.unsqueeze(0)

    return dict(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


class PositionIDFuncCompose:
    def __init__(self, customized_funcs: List[Callable]):
        self.transforms = [base_position_id_func] + customized_funcs

    def __call__(self, **x):
        for t in self.transforms:
            x = t(**x)
        return x


class BaseFoundationModelMixin(PreTrainedModel):
    def get_generation_position_id(self, **kwargs) -> Dict[str, torch.Tensor]:
        """ """
        return None

    def get_position_id_func(self) -> List[Callable]:
        """ """
        return None

    @property
    def position_id_func(self) -> Optional[Callable]:
        if self.get_position_id_func() is None:
            return None
        return PositionIDFuncCompose(self.get_position_id_func())
