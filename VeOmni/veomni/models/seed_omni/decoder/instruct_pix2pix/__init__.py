from transformers import AutoConfig, AutoModel

from .configuring_instruct_pix2pix import InstructPix2PixConfig
from .modeling_instruct_pix2pix import InstructionPix2Pix


AutoConfig.register("instruct_pix2pix", InstructPix2PixConfig)
AutoModel.register(InstructPix2PixConfig, InstructionPix2Pix)
