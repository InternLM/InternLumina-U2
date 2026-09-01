from ....transformers.janus.configuration_janus import JanusVisionConfig
from ..base import BaseEncoderConfigMixin


class JanusSigLIPEncoderConfig(BaseEncoderConfigMixin, JanusVisionConfig):
    model_type = "janussiglip_encoder"

    def __init__(
        self,
        aligner_depth: int = 2,
        aligner_input_dim: int = 1024,
        n_embed: int = 2048,
        aligner_projector_type: str = "mlp_gelu",
        **kwargs,
    ):
        self.aligner_depth = aligner_depth
        self.aligner_input_dim = aligner_input_dim
        self.n_embed = n_embed
        self.aligner_projector_type = aligner_projector_type
        super().__init__(**kwargs)
