from transformers import AutoConfig, AutoModel, AutoProcessor

from .configuring_cosmos import CosmosConfig
from .modeling_cosmos import Cosmos
from .processing_cosmos import CosmosProcessor


AutoConfig.register("cosmos", CosmosConfig)
AutoModel.register(CosmosConfig, Cosmos)
AutoProcessor.register(CosmosConfig, CosmosProcessor)
