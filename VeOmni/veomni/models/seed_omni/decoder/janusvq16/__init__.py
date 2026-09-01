from transformers import AutoConfig, AutoModel, AutoProcessor

from .configuring_janusvq16 import JanusVQ16DecoderConfig
from .modeling_janusvq16 import JanusVQ16Decoder
from .processing_janusvq16 import JanusVQ16DecoderProcessor


AutoConfig.register("janusvq16_decoder", JanusVQ16DecoderConfig)
AutoModel.register(JanusVQ16DecoderConfig, JanusVQ16Decoder)
AutoProcessor.register(JanusVQ16DecoderConfig, JanusVQ16DecoderProcessor)
