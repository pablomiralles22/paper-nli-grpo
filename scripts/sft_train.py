import os
import sys
import hydra
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from dotenv import dotenv_values
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from omegaconf import DictConfig, OmegaConf
from src.nli_prompt_utils import NLIPromptUtils
from src.generalized_completion_only_collator import CompletionOnlyCollator
from src.csv_logger import CSVLogger

# --- Monkey patch to fix multi-GPU device mismatch with num_items_in_batch ---
import transformers.loss.loss_utils
_original_fixed_cross_entropy = transformers.loss.loss_utils.fixed_cross_entropy

def _patched_fixed_cross_entropy(source, target, num_items_in_batch=None, ignore_index=-100, **kwargs):
    if hasattr(num_items_in_batch, "to"):
        num_items_in_batch = num_items_in_batch.to(source.device)
    return _original_fixed_cross_entropy(source, target, num_items_in_batch, ignore_index, **kwargs)

transformers.loss.loss_utils.fixed_cross_entropy = _patched_fixed_cross_entropy
# -----------------------------------------------------------------------------

ENV_CONFIG = dotenv_values(".env")
HF_ACCESS_TOKEN = ENV_CONFIG["HF_WRITE_TOKEN"]

def load_model_and_tokenizer(model_id, model_init_kwargs={}):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_init_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_ACCESS_TOKEN)
    tokenizer.padding_side = "left" # SFT usually prefers right padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def load_data(nli_prompt_utils, data_path, split="train", seed=42):
    dataset = load_dataset(data_path, split=split, token=HF_ACCESS_TOKEN)
    dataset = dataset.shuffle(seed=seed)
    print("Using seed:", seed)

    # Step 1: Filter out -1 labels
    dataset = dataset.filter(lambda item: (item["label"] != -1))

    system_messages = nli_prompt_utils.get_system_prompt()
    
    def process_fn(item):
        # We assume the dataset has 'premise', 'hypothesis', 'explanation', 'label'
        # If 'explanation' is missing, we pass empty string or handle it
        explanation = item.get("explanation", "")
        item_messages = nli_prompt_utils.build_complete_example(
            item["premise"], item["hypothesis"], explanation, item["label"]
        )
        return { 
            "messages": [ *system_messages, *item_messages ],
        }
    
    return dataset.map(process_fn, batched=False, remove_columns=dataset.column_names)

@hydra.main(version_base=None, config_path="../configs/sft", config_name="base")
def main(cfg : DictConfig) -> None:
    cfg = OmegaConf.to_object(cfg)

    model_id_safe = cfg["model_id"].replace("/", "-")
    run_name = cfg.get("run_name", model_id_safe + "-SFT")
    trainer_config = cfg["trainer_config"]
    output_dir = os.path.join(cfg.get("output_dir", f"out/sft_train"), run_name)
    seed = cfg.get("seed", 42)

    # Add the access token to the model init kwargs
    if "model_init_kwargs" in trainer_config:
        trainer_config["model_init_kwargs"]["token"] = HF_ACCESS_TOKEN

    # Load the NLI prompt utils
    nli_prompt_utils = (
        NLIPromptUtils.get_instance() if "prompt_config" not in cfg
        else NLIPromptUtils(cfg["prompt_config"])
    )

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        cfg["model_id"],
        model_init_kwargs=trainer_config.pop("model_init_kwargs", {})
    )
    train_dataset = load_data(nli_prompt_utils, cfg["data_path"], split=cfg["train_split"], seed=seed)
    eval_dataset = load_data(nli_prompt_utils, cfg["data_path"], split=cfg["eval_split"], seed=seed)

    # Load the PEFT config
    if "lora_config" in cfg:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            **cfg.pop("lora_config"),
        )
        is_quantized = any(method in cfg["model_id"].lower() for method in ["awq", "gptq"])
        if is_quantized:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=trainer_config.get("gradient_checkpointing", False),
            )
        if trainer_config.get("gradient_checkpointing", False):
            model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)

    training_args = SFTConfig(
        run_name=run_name,
        output_dir=output_dir,
        **trainer_config,
    )

    # Initialize the collator
    # Note: We need to know the assistant tag for the specific model
    # For Qwen2.5/Qwen2: "<|im_start|>assistant\n"
    assert "Qwen2.5" in cfg["model_id"], "This script currently assumes the model uses '<|im_start|>assistant\\n' as the assistant tag. Please modify the collator initialization if using a different model."
    assistant_begin_template = "<|im_start|>assistant\n" 
    end_of_turn_template = "<|im_end|>\n"

    collator_fn = CompletionOnlyCollator(
        assistant_begin_template=assistant_begin_template,
        end_of_turn_template=end_of_turn_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator_fn,
        args=training_args,
        callbacks=[CSVLogger(output_dir=output_dir)],
    )

    # Train the model
    trainer.train(resume_from_checkpoint=cfg["use_checkpoint"])

    # Save the model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
