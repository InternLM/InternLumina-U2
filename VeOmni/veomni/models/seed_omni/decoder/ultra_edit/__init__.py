from transformers import AutoConfig, AutoModel, AutoProcessor

from .configuring_ultra_edit import UltraEditConfig
from .modeling_ultra_edit import UltraEdit
from .processing_ultra_edit import UltraEditProcessor


AutoConfig.register("ultra_edit", UltraEditConfig)
AutoModel.register(UltraEditConfig, UltraEdit)
AutoProcessor.register(UltraEditConfig, UltraEditProcessor)
