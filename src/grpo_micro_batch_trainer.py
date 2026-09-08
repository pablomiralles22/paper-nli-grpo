import torch
import torch.nn as nn

from typing import Dict, Union, Any
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.utils import selective_log_softmax
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    is_torch_xpu_available,
    is_torch_mlu_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_mps_available,
    is_apex_available,
    DistributedType,
    OptimizerNames,
)

if is_apex_available():
    from apex import amp


class GRPOMicroBatchTrainer(GRPOTrainer):
    def __init__(self, micro_batch_size=1, **kwargs):
        super().__init__(**kwargs)
        self.micro_batch_size = micro_batch_size

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step with micro-batching support.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        total_loss = torch.tensor(0.0, device=self.accelerator.device)

        if inputs is None or len(inputs) == 0:
            return total_loss

        # If all advantages are zero, skip the batch
        if inputs["advantages"].eq(0).all():
            return total_loss

        batch_size = inputs[next(iter(inputs))].size(0)
        num_micro_batches = max(1, batch_size // self.micro_batch_size)
        
        for i in range(num_micro_batches):
            mb_start, mb_end = i * self.micro_batch_size, (i + 1) * self.micro_batch_size
            micro_inputs = {
                key: (val[mb_start:mb_end] if val is not None else None)
                for key, val in inputs.items()
            }
            
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, micro_inputs, num_items_in_batch=num_items_in_batch)
            
            loss = loss / num_micro_batches  # Normalize loss per micro-batch
            total_loss += loss.detach()
            
            if self.args.n_gpu > 1:
                loss = loss.mean()  # Average loss for multi-GPU training
            
            self.accelerator.backward(loss)

        # Memory cleanup
        del inputs, micro_inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()
        
        return total_loss

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
            use_cache=False,
        ).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens
