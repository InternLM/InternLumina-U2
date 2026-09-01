from transformers import LlamaConfig


class LlamaFoundationConfig(LlamaConfig):
    model_type = "llama_foundation"
    sub_configs = {}
