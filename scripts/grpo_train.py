# train_grpo.py
import os
import sys
import hydra

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import load_dataset
from trl import GRPOConfig
from dotenv import dotenv_values
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from omegaconf import DictConfig, OmegaConf
from src.nli_prompt_utils import NLIPromptUtils
from src.grpo_micro_batch_trainer import GRPOMicroBatchTrainer
from src.csv_logger import CSVLogger

os.environ["WANDB_PROJECT"] = "nli"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "end"  # log all model checkpoints

ENV_CONFIG = dotenv_values(".env")
HF_ACCESS_TOKEN = ENV_CONFIG["HF_WRITE_TOKEN"]

def load_model_and_tokenizer(model_id, model_init_kwargs={}):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_init_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_ACCESS_TOKEN)
    # tokenizer.padding_side = "left"
    # tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    return model, tokenizer

def reward(nli_prompt_utils, completion, label, format_score=0.1, prediction_score=0.9):
    if nli_prompt_utils.is_format_valid(completion) is False:
        return 0.0
    return format_score + prediction_score * (nli_prompt_utils.extract_label_id(completion) == label)

def reward_build(nli_prompt_utils, format_score=0.1, prediction_score=0.9):
    def reward_fn(completions, ground_truth, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        return [
            reward(nli_prompt_utils, content, label, format_score, prediction_score)
            for content, label in zip(completion_contents, ground_truth)
        ]
    return reward_fn

def load_data(nli_prompt_utils, data_path, split="train", seed=42):
    dataset = load_dataset(data_path, split=split, token=HF_ACCESS_TOKEN)
    dataset = dataset.shuffle(seed=seed)
    print("Using seed:", seed)

    # Step 1: Filter out -1 labels
    dataset = dataset.filter(lambda item: (item["label"] != -1))

    system_messages = nli_prompt_utils.get_system_prompt()
    
    def process_fn(item):
        item_messages = nli_prompt_utils.build_incomplete_example(item["premise"], item["hypothesis"])
        return { 
            "prompt": [ *system_messages, *item_messages, ],
            "ground_truth": item["label"],
        }
    
    return dataset.map(process_fn, batched=False, remove_columns=dataset.column_names)

@hydra.main(version_base=None, config_path="../configs/grpo", config_name="base")
def main(cfg : DictConfig) -> None:
    cfg = OmegaConf.to_object(cfg)

    model_id_safe = cfg["model_id"].replace("/", "-")
    run_name = cfg.get("run_name", model_id_safe + "-GRPO")
    trainer_config = cfg["trainer_config"]
    output_dir = os.path.join(cfg.get("output_dir", f"out/rl_train"), run_name)
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
    dataset = load_data(nli_prompt_utils, cfg["data_path"], split=cfg["data_split"], seed=seed)

    # Load the PEFT config
    if "lora_config" in cfg:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # Adjust for different architectures
            **cfg.pop("lora_config"),
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)


    training_args = GRPOConfig(
        run_name=run_name,
        output_dir=output_dir,
        **trainer_config,
    )

    trainer = GRPOMicroBatchTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_build(nli_prompt_utils),
        args=training_args,
        train_dataset=dataset,
        micro_batch_size=cfg["micro_batch_size"],
        callbacks=[CSVLogger(output_dir=output_dir)],
    )
    trainer.train(resume_from_checkpoint=cfg["use_checkpoint"])


if __name__ == "__main__":
    main()
