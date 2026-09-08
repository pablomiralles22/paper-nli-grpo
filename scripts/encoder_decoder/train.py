import torch
import argparse
import os
import yaml
import sys
import numpy as np
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from functools import partial
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset, load_from_disk
from dotenv import dotenv_values

os.environ["WANDB_PROJECT"] = "nli"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"  # don't log the model to W&B

ENV_CONFIG = dotenv_values(".env")
HF_ACCESS_TOKEN = ENV_CONFIG["HF_WRITE_TOKEN"]


def load_model_and_tokenizer(model_id):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"

    return model, tokenizer

INPUT_TEMPLATE = "Is this true and why? {premise} implies {hypothesis}"
OUTPUT_TEMPLATE = "{affirmation} it is {label} because {explanation}"
ID_TO_LABEL = { 0: "entailment", 1: "neutral", 2: "contradiction" }

def load_data(data_path, tokenizer):
    if os.path.exists(data_path):
        dataset = load_from_disk(data_path)
    else:
        dataset = load_dataset(data_path, token=HF_ACCESS_TOKEN)

    dataset = dataset.filter(
        lambda items: [label != -1 for label in items["label"]], 
        batched=True, num_proc=8
    )

    # Step 1: Add messages
    def process_fn_1(item):
        label_id = item["label"]
        label = ID_TO_LABEL[label_id]
        affirmation = ("Yes" if label_id == 0 else "No")
        encoder_input =  INPUT_TEMPLATE.format(premise=item["premise"], hypothesis=item["hypothesis"])
        decoder_output = OUTPUT_TEMPLATE.format(affirmation=affirmation, label=label, explanation=item.get("explanation", ""))
        return {
            "input": encoder_input, 
            "output": decoder_output,
            "label_id": label_id
        }
    column_names = list(dataset.values())[0].column_names
    dataset = dataset.map(
        process_fn_1, batched=False, remove_columns=column_names, num_proc=8
    )

    # Step 2: Tokenize the data
    def process_fn_2(batch):
        return {
            "input_ids": tokenizer(batch["input"], padding=False, truncation=True, max_length=512)["input_ids"],
            "labels": tokenizer(batch["output"], padding=False, truncation=True, max_length=128)["input_ids"],
            "label_id": batch["label_id"],
        }
    column_names = list(dataset.values())[0].column_names
    dataset = dataset.map(
        process_fn_2, batched=True, remove_columns=column_names, num_proc=8
    )

    return dataset

def compute_metrics(tokenizer, binary_mode, eval_pred):
    predictions, labels, inputs = eval_pred.predictions, eval_pred.label_ids, eval_pred.inputs

    # Change -100 values to padding token
    predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

    # Decode the predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Extract the NLI labels
    REGEX = re.compile(r"^(yes|no) it is (entailment|contradiction|neutral) because(.*)?$", re.IGNORECASE)
    def get_matches(pred):
        match = REGEX.match(pred.lower())
        if match:
            affirmation, label, _ = match.groups()
            return affirmation, label
        return None, None
        
    pred_matches = [get_matches(pred) for pred in decoded_preds]
    true_matches = [get_matches(label) for label in decoded_labels]

    if binary_mode is False:
        nli_labels = ["entailment", "contradiction", "neutral"]
        pred_labels = [match[1] for match in pred_matches]
        true_labels = [match[1] for match in true_matches]
    else:
        nli_labels = ["yes", "no"]
        pred_labels = [match[0] for match in pred_matches]
        true_labels = [match[0] for match in true_matches]
    
    # Calculate accuracy
    correct = sum(p == t for p, t in zip(pred_labels, true_labels) if p is not None and t is not None)
    total = sum(1 for t in true_labels if t is not None)
    
    accuracy = correct / total if total > 0 else 0
    
    # Calculate label-specific metrics
    label_metrics = {}
    for label in nli_labels:
        tp = sum(1 for p, t in zip(pred_labels, true_labels) if p == label and t == label)
        fp = sum(1 for p, t in zip(pred_labels, true_labels) if p == label and t != label)
        fn = sum(1 for p, t in zip(pred_labels, true_labels) if p != label and t == label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        label_metrics[f"{label}_precision"] = precision
        label_metrics[f"{label}_recall"] = recall
        label_metrics[f"{label}_f1"] = f1
    
    metrics = {
        "accuracy": accuracy,
        "label_extraction_rate": sum(1 for p in pred_labels if p is not None) / len(pred_labels),
        **label_metrics
    }
    
    return metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model_id", type=str, default="google-t5/t5-3b")
    parser.add_argument("--output_model_id", type=str, required=False)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="pablomiralles22/esnli")
    parser.add_argument("--trainer_config", type=str, default="configs/encoder_decoder/base.yaml")

    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--binary_mode", action="store_true")

    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--test_split", type=str, default="test")

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
    
    output_dir = f"out/{run_name}"

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(input_model_id)
    
    # Load the training dataset
    dataset = load_data(data_path, tokenizer)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        **trainer_config,
        report_to="wandb",
        run_name=run_name,
        predict_with_generate=True,
    )

    collator_fn = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Initialize the trainer with compute_metrics
    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=dataset[args.train_split],
        eval_dataset=dataset[args.val_split],
        processing_class=tokenizer,
        data_collator=collator_fn,
        args=training_args,
        compute_metrics=partial(compute_metrics, tokenizer, args.binary_mode),
    )

    # Train the model
    if not args.test_only:
        trainer.train()

    # Test the model
    trainer.evaluate(dataset[args.test_split])

    # upload the model to the hub
    if output_model_id is not None:
        model.push_to_hub(output_model_id, token=HF_ACCESS_TOKEN, private=True)
        tokenizer.push_to_hub(output_model_id, token=HF_ACCESS_TOKEN, private=True)

if __name__ == "__main__":
    main()
