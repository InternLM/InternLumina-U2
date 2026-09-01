from typing import Tuple, Union

import torch
from transformers import SeedOssForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..base import BaseFoundationModelMixin
from .configuration_seed_oss_foundation import SeedOssFoundationConfig


class SeedOssFoundationModel(BaseFoundationModelMixin, SeedOssForCausalLM):
    config_class = SeedOssFoundationConfig

    def __init__(self, config: SeedOssFoundationConfig, **kwargs):
        super().__init__(config, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        position_ids: torch.LongTensor = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if position_ids is not None and position_ids.ndim == 3:
            position_ids = position_ids.squeeze(1)  # bs, 1, l -> bs, l
        if inputs_embeds is not None:
            return super().forward(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                **kwargs,
            )
        else:
            return super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                **kwargs,
            )
