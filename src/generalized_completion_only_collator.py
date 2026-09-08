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
        assistant_begin_template: str,
        end_of_turn_template: str,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.assistant_begin_template = assistant_begin_template
        self.end_of_turn_template = end_of_turn_template

        self.assistant_begin_template_token_ids = self.tokenizer.encode(self.assistant_begin_template, add_special_tokens=False)
        self.end_of_turn_template_token_ids = self.tokenizer.encode(self.end_of_turn_template, add_special_tokens=False)

        if type(self.assistant_begin_template_token_ids) is not list:
            self.assistant_begin_template_token_ids = [self.assistant_begin_template_token_ids]
        if type(self.end_of_turn_template_token_ids) is not list:
            self.end_of_turn_template_token_ids = [self.end_of_turn_template_token_ids]

        self.ignore_index = ignore_index

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            assistant_starts = self.__get_starts(batch["labels"][i], self.assistant_begin_template_token_ids)

            if len(assistant_starts) == 0:
                warnings.warn(
                    f"Could not find assistant begin key `{self.assistant_begin_template}` in the following instance: "
                    f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss calculation. Note, if this happens often, consider increasing the `max_seq_length`.",
                    UserWarning,
                )
                batch["labels"][i, :] = self.ignore_index
                continue

            end_of_turn_starts = [
                idx + len(self.end_of_turn_template_token_ids)
                for idx in self.__get_starts(batch["labels"][i], self.end_of_turn_template_token_ids)
            ]
            end_of_turn_starts.extend(self.__get_starts(batch["labels"][i], [self.tokenizer.eos_token_id]))
            end_of_turn_starts.append(len(batch["labels"][i]))

            matched_intervals = []
            for assistant_start in assistant_starts:
                end_of_turn_starts_after_assistant_start = [idx for idx in end_of_turn_starts if idx > assistant_start]
                if len(end_of_turn_starts_after_assistant_start) == 0:
                    warnings.warn(
                        f"Could not find end of turn key `{self.end_of_turn_template}` or EOS token after assistant start in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. The tokens after the assistant start will be included in loss calculation. Note, if this happens often, consider increasing the `max_seq_length`.",
                        UserWarning,
                    )
                    matched_intervals.append((assistant_start, len(batch["labels"][i])))
                else:
                    matched_intervals.append((assistant_start, min(end_of_turn_starts_after_assistant_start)))

            prev_end = 0
            for start, end in matched_intervals:
                batch["labels"][i, prev_end:start] = self.ignore_index
                prev_end = end
            
            if prev_end < len(batch["labels"][i]):
                batch["labels"][i, prev_end:] = self.ignore_index

        return batch

    def __get_starts(self, labels, response_token_ids) -> list[int]:
        starts = []
        for idx in np.where(labels == response_token_ids[0])[0]:
            if (response_token_ids == labels[idx : idx + len(response_token_ids)].tolist()):
                starts.append(idx)
        return starts
