import torch

from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling

class MultitaskCollatorFn:
    def __init__(self, tokenizer, apply_aux=True, mlm_probability=0.3):
        self.tokenizer = tokenizer
        self.apply_aux = apply_aux
        self.mlm_probability = mlm_probability
        self.padding_collator = DataCollatorWithPadding(tokenizer)
        # self.lm_data_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=mlm_probability)

    def __call__(self, batch):
        classification_batch = [self._get_subtask_item(item, 'classification') for item in batch]
        class_batch = self.padding_collator(classification_batch)

        if self.apply_aux is False:
            return {'classification': class_batch}

        aux_batch = [self._get_subtask_item(item, 'aux') for item in batch]
        aux_batch = self.padding_collator(aux_batch)
        aux_batch = self._mask_only_third_sentence(aux_batch)

        return {
            'classification': class_batch,
            'aux': aux_batch,
        }

    def _mask_only_third_sentence(self, inputs):
        input_ids = inputs["input_ids"]

        third_sentence_mask = ((input_ids == self.tokenizer.sep_token_id).cumsum(dim=1) >= 2)
        mlm_mask = torch.rand(input_ids.shape) < self.mlm_probability
        mlm_mask &= third_sentence_mask
        mlm_mask &= (input_ids != self.tokenizer.sep_token_id)  # Do not mask [SEP] tokens
        mlm_mask &= (input_ids != self.tokenizer.pad_token_id)  # Do not mask [PAD] tokens

        # Create labels
        labels = input_ids.clone()
        labels[~mlm_mask] = -100
        
        # Apply mask
        input_ids[mlm_mask] = self.tokenizer.mask_token_id
        
        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }

    def _get_subtask_item(self, item, prefix):
        subtask_item = dict()
        for key, val in item.items():
            if key.startswith(prefix) is False:
                continue
            key_wo_prefix = key[len(prefix) + 1:]
            subtask_item[key_wo_prefix] = val
        return subtask_item

    # def _mask_only_third_sentence(self, inputs):
    #     input_ids = inputs["input_ids"]
    #     labels = inputs["labels"]
        
    #     sep_token_id = self.tokenizer.sep_token_id  # ID of the [SEP] token
    #     batch_size, seq_length = input_ids.shape

    #     for i in range(batch_size):
    #         sep_indices = (input_ids[i] == sep_token_id).nonzero(as_tuple=True)[0]

    #         if len(sep_indices) < 2:
    #             continue  # Skip if the sequence does not have two [SEP] tokens

    #         second_sep_idx = sep_indices[1].item()  # Position of the second [SEP]

    #         # Restore original tokens in input_ids where we remove masks
    #         mask_positions = (labels[i] != -100) & (torch.arange(seq_length) <= second_sep_idx)
    #         input_ids[i, mask_positions] = labels[i, mask_positions]

    #         # Mask out labels before the third sentence
    #         labels[i, :second_sep_idx + 1] = -100  # Keep only masks after the second [SEP]

    #     return inputs