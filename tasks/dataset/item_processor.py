from abc import ABC, abstractmethod
from typing import List, Tuple


class ItemProcessorBase(ABC):
    """Interface every dataset item processor implements.

    ``process_item`` turns one raw annotation record into the tensors the
    collator expects.  ``predict_item_token_length`` is consulted by the
    length-aware packer before the record is actually loaded, so it must be
    cheap and must not under-estimate -- an under-estimate shows up later as an
    oversize pack that gets skipped.
    """

    @abstractmethod
    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        raise NotImplementedError

    def predict_item_token_length(self, data_item: dict) -> int:
        return 1
