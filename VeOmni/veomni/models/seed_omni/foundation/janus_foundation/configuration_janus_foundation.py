from transformers import LlamaConfig


class JanusFoundationConfig(LlamaConfig):
    model_type = "janus_foundation"
    sub_configs = {}
