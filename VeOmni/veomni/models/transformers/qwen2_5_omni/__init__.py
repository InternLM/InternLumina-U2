from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
)

from ....utils.import_utils import is_transformers_version_greater_or_equal_to
from .configuration_qwen2_5_omni import Qwen2_5OmniConfig
from .modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from .processing_qwen2_5_omni import Qwen2_5OmniProcessor


# After 4.52, this model is already registered in transfomers. Register will cause
# already exists error.
if not is_transformers_version_greater_or_equal_to("4.52.0"):
    AutoConfig.register("qwen2_5_omni", Qwen2_5OmniConfig)
    AutoModel.register(Qwen2_5OmniConfig, Qwen2_5OmniForConditionalGeneration)
    AutoModelForCausalLM.register(Qwen2_5OmniConfig, Qwen2_5OmniForConditionalGeneration)
    AutoProcessor.register(Qwen2_5OmniConfig, Qwen2_5OmniProcessor)
