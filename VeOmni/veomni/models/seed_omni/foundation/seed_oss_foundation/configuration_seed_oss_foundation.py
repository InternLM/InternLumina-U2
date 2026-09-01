from ..base import BaseFoundationConfigMixin


class SeedOssFoundationConfig(BaseFoundationConfigMixin):
    model_type = "seed_oss_foundation"

    def __init__(self, vocab_size: int = 0, hidden_size: int = 0, tie_word_embeddings: bool = None, **kwargs):
        super().__init__(vocab_size, hidden_size, tie_word_embeddings, **kwargs)
