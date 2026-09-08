import argparse
import ast
import json
import os
import sys
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
from dotenv import dotenv_values
from peft import AutoPeftModelForSequenceClassification, PeftModel
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


ENV_CONFIG = dotenv_values(".env")
HF_ACCESS_TOKEN = ENV_CONFIG.get("HF_WRITE_TOKEN")


def parse_args():
    none_or_str = lambda value: None if value == "None" else value

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=False)
    parser.add_argument("--model-id", type=none_or_str, default=None)
    parser.add_argument("--lora-adapter-path", type=none_or_str, default=None)
    parser.add_argument("--data-path", type=str, default="snli")
    parser.add_argument("--data-name", type=none_or_str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--premise-column", type=str, default="premise")
    parser.add_argument("--hypothesis-column", type=str, default="hypothesis")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binary-mode", type=ast.literal_eval, default=False)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--attn-implementation", type=none_or_str, default=None)
    parser.add_argument("--output-file", type=str, required=True)
    return parser.parse_args()


def resolve_torch_dtype(dtype):
    if dtype == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype}'. Use one of: auto, float16, bfloat16, float32.")
    return mapping[dtype]


def load_model_and_tokenizer(
    model_id,
    num_labels=3,
    torch_dtype="auto",
    attn_implementation=None,
    lora_adapter_path=None,
):
    model_kwargs = {"token": HF_ACCESS_TOKEN, "torch_dtype": torch_dtype, "num_labels": num_labels}
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    if lora_adapter_path is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_id, **model_kwargs)
        tokenizer_id = model_id
    elif model_id is None:
        model = AutoPeftModelForSequenceClassification.from_pretrained(lora_adapter_path, **model_kwargs)
        tokenizer_id = model.peft_config["default"].base_model_name_or_path
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(model_id, **model_kwargs)
        model = PeftModel.from_pretrained(base_model, lora_adapter_path, token=HF_ACCESS_TOKEN)
        tokenizer_id = model_id

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, token=HF_ACCESS_TOKEN)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                f"Tokenizer for '{model_id}' has no pad_token/eos_token. "
                "Set a pad token before evaluation."
            )
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id

    if attn_implementation == "flash_attention_2":
        tokenizer.padding_side = "left"

    return model, tokenizer


def load_data(
    tokenizer,
    data_path,
    split,
    data_name=None,
    premise_column="premise",
    hypothesis_column="hypothesis",
    label_column="label",
    max_length=1024,
):
    dataset = load_dataset(data_path, name=data_name, split=split, token=HF_ACCESS_TOKEN)
    dataset = dataset.filter(lambda item: item[label_column] != -1)
    dataset = dataset.shuffle(seed=42)

    def preprocess_fn(batch):
        tokenized = tokenizer(
            [f"Premise: {item}" for item in batch[premise_column]],
            [f"Hypothesis: {item}" for item in batch[hypothesis_column]],
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = batch[label_column]
        return tokenized

    return dataset.map(preprocess_fn, batched=True, remove_columns=dataset.column_names)


def calculate_metrics(y_true, y_pred, labels):
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        zero_division=0,
    )
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)

    metrics = {
        "accuracy": float(accuracy),
        "class_metrics": {},
        "confusion_matrix": conf_matrix.tolist(),
    }

    for i, label in enumerate(labels):
        metrics["class_metrics"][str(label)] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }

    return metrics


def main():
    args = parse_args()
    if args.model_id is None and args.lora_adapter_path is None:
        raise ValueError("Provide --model-id (full model) or --lora-adapter-path (PEFT adapter).")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        torch_dtype=resolve_torch_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        lora_adapter_path=args.lora_adapter_path,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    dataset = load_data(
        tokenizer=tokenizer,
        data_path=args.data_path,
        split=args.split,
        data_name=args.data_name,
        premise_column=args.premise_column,
        hypothesis_column=args.hypothesis_column,
        label_column=args.label_column,
        max_length=args.max_length,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

            y_true.extend(labels.tolist())
            y_pred.extend(preds)

    label_names = ["entailment", "neutral", "contradiction"]
    if args.binary_mode:
        label_names = ["entailment", "not-entailment"]
        y_true = [(0 if label == 0 else 1) for label in y_true]
        y_pred = [(0 if pred == 0 else 1) for pred in y_pred]

    metrics = calculate_metrics(y_true, y_pred, labels=label_names)

    output_data = {
        "params": vars(args),
        "results": metrics,
    }

    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    main()
