from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from transformers import (
    __version__ as transformers_version,
)

from ....utils.import_utils import is_transformers_version_greater_or_equal_to
from .configuration_janus import JanusConfig
from .image_processing_janus import JanusImageProcessor
from .modeling_janus import Janus
from .processing_janus import JanusChatProcessor


# After 4.52, this model is already registered in transfomers. Register will cause
# already exists error.
if not is_transformers_version_greater_or_equal_to("4.52.0"):
    AutoConfig.register("janus", JanusConfig)
    AutoModel.register(JanusConfig, Janus)
    AutoModelForVision2Seq.register(JanusConfig, Janus)
    AutoProcessor.register(JanusConfig, JanusChatProcessor)
    AutoImageProcessor.register(JanusConfig, JanusImageProcessor)
