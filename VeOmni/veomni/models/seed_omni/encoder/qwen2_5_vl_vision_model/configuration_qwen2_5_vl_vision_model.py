from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

from ..base import BaseEncoderConfigMixin


class Qwen25VLVisionModelConfig(BaseEncoderConfigMixin, Qwen2_5_VLVisionConfig):
    model_type = "qwen2_5_vl_vision_model"

    def __init__(
        self,
        return_hidden_states=False,
        train_origin_projector=False,
        **kwargs,
    ):
        self.return_hidden_states = return_hidden_states
        self.train_origin_projector = train_origin_projector
        super().__init__(**kwargs)
