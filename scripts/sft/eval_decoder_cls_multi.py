import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from safetensors import safe_open
from src.nli_prompt_utils import NLIPromptUtils
from dotenv import dotenv_values

os.environ["WANDB_PROJECT"] = "nli"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "end"  # log all model checkpoints

ENV_CONFIG = dotenv_values(".env")
HF_ACCESS_TOKEN = ENV_CONFIG["HF_WRITE_TOKEN"]

CLS_TOKEN = "<|cls|>"

def calculate_accuracy(model, input_ids, attention_mask, cls_token_id, cls_labels):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        lhs = outputs.hidden_states[-1]
        cls_token_mask = (input_ids == cls_token_id)
        cls_embeddings = lhs[cls_token_mask]
        cls_logits = model.cls_head(cls_embeddings)
        cls_preds = cls_logits.argmax(dim=-1)
        accuracy = (cls_preds == cls_labels).float().mean()
    return accuracy

def load_cls_head_weights(model_id):
    files = [f for f in os.listdir(model_id) if f.endswith(".safetensors")]
    file_paths = [os.path.join(model_id, f) for f in files]
    tensors = {}

    for file_path in file_paths:
        with safe_open(file_path, framework="pt") as f:
            if "cls_head.bias" in f.keys():
                tensors["bias"] = f.get_tensor("cls_head.bias")
            if "cls_head.weight" in f.keys():
                tensors["weight"] = f.get_tensor("cls_head.weight")
                
    return tensors

def load_model_and_tokenizer(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # tokenizer.add_tokens(CLS_TOKEN)
    # tokenizer.padding_side = "left"

    # model.resize_token_embeddings(len(tokenizer))
    model.cls_head = nn.Linear(model.config.hidden_size, 3)
    tensors = load_cls_head_weights(model_id)
    model.cls_head.load_state_dict(tensors)
    model.cls_head = model.cls_head.to("cuda:0", dtype=torch.bfloat16)

    return model, tokenizer

def load_data(nli_prompt_utils, tokenizer, data_path, split="train"):
    dataset = load_dataset(data_path, split=split, token=HF_ACCESS_TOKEN)
    dataset = dataset.shuffle(seed=42)

    # Step 1: Filter out incorrect predictions, and remove -1 labels
    def filter_fn(item):
        if item["label"] == -1: return False
        return True

    dataset = dataset.filter(filter_fn)

    system_messages = nli_prompt_utils.get_system_prompt()

    # Step 2: Add messages
    def process_fn(item):
        item_messages = nli_prompt_utils.build_incomplete_example(item["premise"], item["hypothesis"])
        item_messages[0]["content"] = item_messages[0]["content"] + CLS_TOKEN
        return { 
            "messages": [*system_messages, *item_messages],
            "classification_label": item["label"],
        }
        
    dataset = dataset.map(
        process_fn,
        batched=False,
        remove_columns=dataset.column_names,
    )
    
    # Step 3: apply chat template
    def chat_fn(items):
        tokenizer_output = tokenizer.apply_chat_template(
            items["messages"],
            tokenize=True,
            padding=False,
            truncation=False,
        )
        return {"input_ids": tokenizer_output}
    dataset = dataset.map(chat_fn, batched=True)
    
    return dataset

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model_id", type=str)
    parser.add_argument("--data_path", type=str, default="anli")
    parser.add_argument("--data_split", type=str, default="test_r3")
    parser.add_argument("--prompt_config", type=str, default="configs/prompts/deepseek.yaml")

    return parser.parse_args()

def main():
    args = parse_args()
    args_dict = vars(args)

    # Unpack the arguments
    input_model_id = args_dict.pop("input_model_id")
    data_path = args_dict.pop("data_path")
    data_split = args_dict.pop("data_split")

    nli_prompt_utils = (
        NLIPromptUtils.get_instance() if args.prompt_config is None
        else NLIPromptUtils(args.prompt_config)
    )

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(input_model_id)
    model.eval()
    cls_token_id = tokenizer.convert_tokens_to_ids(CLS_TOKEN)

    # Load the training dataset
    dataset = load_data(nli_prompt_utils, tokenizer, data_path, data_split)

    # Evaluate the model
    accuracies = []
    batch_size = 2
    for idx in range(0, len(dataset), batch_size):
        batch = dataset[idx:idx + batch_size]
        padded_batch = tokenizer.pad({"input_ids": batch["input_ids"]}, return_tensors="pt")
        padded_batch = {k: v.to("cuda:0") for k, v in padded_batch.items()}
        cls_labels = torch.tensor([batch["classification_label"]]).to("cuda:0")

        accuracy = calculate_accuracy(model, padded_batch["input_ids"], padded_batch["attention_mask"], cls_token_id, cls_labels)
        accuracies.append(accuracy.item())
        print(f"Accuracy: {np.mean(accuracies)}")


if __name__ == "__main__":
    main()
