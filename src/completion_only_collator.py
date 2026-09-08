import warnings
import torch
import numpy as np

from typing import Any, Union
from transformers import DataCollatorForLanguageModeling

class CompletionOnlyCollator(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, list[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        mlm (`bool`, *optional*, defaults to `False`): Whether to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_templates: list[str],
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.response_templates = response_templates
        self.response_token_ids_list = [
            self.tokenizer.encode(response_template, add_special_tokens=False)
            for response_template in response_templates
        ]

        self.ignore_index = ignore_index

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            response_token_ids_start_idxs = [
                self.__get_response_token_ids_start_idx(batch["labels"][i], response_token_ids)
                for response_token_ids in self.response_token_ids_list
            ]
            response_token_ids_start_idxs = [idx for idx in response_token_ids_start_idxs if idx is not None]


            if len(response_token_ids_start_idxs) == 0:
                warnings.warn(
                    f"Could not find response key `{self.response_templates}` in the following instance: "
                    f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss "
                    "calculation. Note, if this happens often, consider increasing the `max_seq_length`.",
                    UserWarning,
                )
                batch["labels"][i, :] = self.ignore_index
            else:
                response_token_ids_start_idx = max(response_token_ids_start_idxs)
                batch["labels"][i, :response_token_ids_start_idx] = self.ignore_index
        
        return batch

    def __get_response_token_ids_start_idx(self, labels, response_token_ids) -> int:
        for idx in np.where(labels == response_token_ids[0])[0]:
            if (response_token_ids == labels[idx : idx + len(response_token_ids)].tolist()):
                return idx
        return None
