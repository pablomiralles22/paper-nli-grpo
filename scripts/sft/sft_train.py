import torch
import argparse
import os
import yaml
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from src.nli_prompt_utils import NLIPromptUtils
from src.completion_only_collator import CompletionOnlyCollator
from dotenv import dotenv_values

os.environ["WANDB_PROJECT"] = "nli"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "end"  # log all model checkpoints

ENV_CONFIG = dotenv_values(".env")
HF_ACCESS_TOKEN = ENV_CONFIG["HF_WRITE_TOKEN"]

def load_model_and_tokenizer(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "left"

    return model, tokenizer

def load_data(nli_prompt_utils, data_path):
    dataset = load_dataset(data_path, split="train", token=HF_ACCESS_TOKEN)
    dataset = dataset.shuffle(seed=42)

    # Step 1: Filter out incorrect predictions, and remove -1 labels
    def filter_fn(item):
        if item["label"] == -1: return False
        if item["label"] != item["predicted_label"]: return False
        return True

    dataset = dataset.filter(filter_fn)

    system_messages = nli_prompt_utils.get_system_prompt()

    # Step 2: Add messages
    def process_fn(item):
        item_messages = nli_prompt_utils.build_complete_example(
            item["premise"], item["hypothesis"], item["explanation"], item["label"]
        )
        return { 
            "messages": [*system_messages, *item_messages]
        }
    dataset = dataset.map(
        process_fn,
        batched=False,
        remove_columns=["premise", "hypothesis", "explanation", "label"],
    )
    
    return dataset

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_model_id", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="pablomiralles22/snli-bootstrap-qwen-self-verify")

    parser.add_argument("--trainer_config", type=str, default="configs/sft/sft.yaml")
    parser.add_argument("--prompt_config", type=str, default="configs/prompts/deepseek.yaml")

    return parser.parse_args()

def main():
    args = parse_args()
    args_dict = vars(args)

    # Unpack the arguments
    input_model_id = args_dict.pop("input_model_id")
    output_model_id = args_dict.pop("output_model_id")
    run_name = args_dict.pop("run_name")
    data_path = args_dict.pop("data_path")

    # Load the trainer config
    with open(args_dict.pop("trainer_config"), "r") as f:
        trainer_config = yaml.safe_load(f)
    
    output_dir = f"out/{output_model_id}-{run_name}"

    nli_prompt_utils = (
        NLIPromptUtils.get_instance() if args.prompt_config is None
        else NLIPromptUtils(args.prompt_config)
    )

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(input_model_id)
    model.train()

    # Load the training dataset
    dataset = load_data(nli_prompt_utils, data_path)

    # Define training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        **trainer_config,
        group_by_length=True,
        dataset_text_field="text",
        packing=False,
        report_to="wandb",
        run_name=run_name,
    )

    # Initialize the trainer
    collator_fn = CompletionOnlyCollator(
        # response_templates= ["<|im_start|>assistant\n", "Wait, I think I have made a mistake"],
        response_templates= ["<|im_start|>assistant\n"],
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator_fn,
        args=training_args,
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")
    
    # upload the model to the hub
    model.push_to_hub(output_model_id, token=HF_ACCESS_TOKEN, private=True)
    tokenizer.push_to_hub(output_model_id, token=HF_ACCESS_TOKEN, private=True)

if __name__ == "__main__":
    main()
