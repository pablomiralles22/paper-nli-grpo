import torch
import argparse
import yaml
import ast
import multiprocessing as mp
import sys
import os
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import dotenv_values
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from vllm.lora.request import LoRARequest
from src.nli_prompt_utils import NLIPromptUtils

# Load environment variables
ENV_CONFIG = dotenv_values(".env")
HF_ACCESS_TOKEN = ENV_CONFIG["HF_WRITE_TOKEN"]

def load_pipeline(model_id, dtype="auto", is_lora_enabled=False, quantization=None, max_lora_rank=64):
    num_available_gpus = torch.cuda.device_count()
    return LLM(
        model=model_id,
        dtype=dtype,
        tensor_parallel_size=num_available_gpus,
        max_model_len=1024,
        enable_prefix_caching=True,
        quantization=quantization,
        gpu_memory_utilization=0.7,
        enable_lora=is_lora_enabled,
        max_lora_rank=max_lora_rank,
        enforce_eager=True,
    )

def load_data(nli_prompt_utils, data_path, split="test", data_name=None):
    dataset = load_dataset(data_path, name=data_name, split=split, token=HF_ACCESS_TOKEN)
    dataset = dataset.filter(lambda item: item["label"] != -1)
    dataset = dataset.shuffle(seed=42)

    system_messages = nli_prompt_utils.get_system_prompt()
    
    def process_fn(item):
        item_messages = nli_prompt_utils.build_incomplete_example(item["premise"], item["hypothesis"])
        return { 
            **item,
            "messages": [
                *system_messages,
                *item_messages,
            ]
        }
    
    return dataset.map(process_fn, batched=False)

def extract_label(nli_prompt_utils, output):
    texts = [o.text for o in output.outputs]
    labels = [nli_prompt_utils.extract_label_id(text) for text in texts]
    return labels

def calculate_metrics(y_true, y_pred_list, labels):
    """Calculate mean and std of precision, recall, f1 for each class and overall accuracy across multiple predictions."""
    metrics_list = []
    conf_matrices = []
    
    # Calculate metrics for each prediction set
    for y_pred in y_pred_list:
        conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        conf_matrices.append(conf_matrix)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2])
        accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
        
        pred_metrics = {
            "accuracy": float(accuracy),
            "class_metrics": {}
        }
        
        for i, label in enumerate(labels):
            pred_metrics["class_metrics"][str(label)] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i])
            }
        
        metrics_list.append(pred_metrics)
    
    # Calculate mean and std for each metric
    result_metrics = {
        "accuracy": {
            "mean": float(np.mean([m["accuracy"] for m in metrics_list])),
            "std": float(np.std([m["accuracy"] for m in metrics_list])),
        },
        "class_metrics": {}
    }
    
    for label in labels:
        label_str = str(label)
        result_metrics["class_metrics"][label_str] = {
            "precision": {
                "mean": float(np.mean([m["class_metrics"][label_str]["precision"] for m in metrics_list])),
                "std": float(np.std([m["class_metrics"][label_str]["precision"] for m in metrics_list])),
            },
            "recall": {
                "mean": float(np.mean([m["class_metrics"][label_str]["recall"] for m in metrics_list])),
                "std": float(np.std([m["class_metrics"][label_str]["recall"] for m in metrics_list])),
            },
            "f1": {
                "mean": float(np.mean([m["class_metrics"][label_str]["f1"] for m in metrics_list])),
                "std": float(np.std([m["class_metrics"][label_str]["f1"] for m in metrics_list])),
            }
        }
    
    return result_metrics, conf_matrices

    
def parse_args():
    none_or_str = lambda value: None if value == "None" else value

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str)

    parser.add_argument("--model-id", type=str)
    parser.add_argument("--lora-adapter-path", type=none_or_str, required=False)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--quantization", type=none_or_str, required=False)
    parser.add_argument("--dtype", type=str, default="auto")

    parser.add_argument("--prompt-config", type=str, default="configs/prompts/deepseek.yaml")
    parser.add_argument("--temperature", type=float, default=0.9)

    parser.add_argument("--data-path", type=str, default="snli")
    parser.add_argument("--data-name", type=none_or_str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--binary-mode", type=ast.literal_eval, default=False)

    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-generations", type=int, default=1)

    parser.add_argument("--output-file", type=str, required=False)

    return parser.parse_args()

def main():
    args = parse_args()

    if args.output_file is not None:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Set prompt config
    nli_prompt_utils = NLIPromptUtils(args.prompt_config) 

    # Load model and tokenizer
    is_lora_enabled = args.lora_adapter_path is not None
    llm = load_pipeline(
        model_id=args.model_id,
        dtype=args.dtype,
        is_lora_enabled=is_lora_enabled,
        quantization=args.quantization,
        max_lora_rank=args.max_lora_rank,
    )
    dataset = load_data(nli_prompt_utils, args.data_path, args.split, args.data_name)
    
    sampling_params = SamplingParams(
        max_tokens=768,
        top_p=0.9,
        temperature=args.temperature,
        n=args.num_generations,
    )

    lora_request = None
    if is_lora_enabled is True:
        lora_request = LoRARequest(
            lora_name="lora_adapter",
            lora_int_id=1,
            lora_path=args.lora_adapter_path,
            base_model_name=args.model_id,
        )

    y_true = []
    y_pred_list = [[] for _ in range(args.num_generations)]
    
    for idx in tqdm(range(0, len(dataset), args.batch_size)):
        batch = dataset[idx:idx + args.batch_size]
        outputs = llm.chat(batch["messages"], sampling_params, lora_request=lora_request, use_tqdm=False)
        
        for label, output in zip(batch["label"], outputs):
            y_true.append(label)

            predicted_labels = extract_label(nli_prompt_utils, output)
            
            for y_pred, predicted_label in zip(y_pred_list, predicted_labels):
                y_pred.append(predicted_label)
        
        # for messages, label, output in zip(batch["messages"], batch["label"], outputs):
        #     log_dict = {"prompt": messages, "label": label, "output": output.outputs[0].text}
        #     with open(args.output_file.replace(".json", "_log.jsonl"), "a") as f:
        #         f.write(json.dumps(log_dict) + "\n")
    
    # Calculate metrics and confusion matrix
    labels = ["entailment", "neutral", "contradiction"]
    if args.binary_mode:
        labels = ["entailment", "not-entailment"]
        y_true = [(0 if label == 0 else 1) for label in y_true]
        y_pred_list = [[(0 if pred == 0 else 1) for pred in preds] for preds in y_pred_list]

    metrics, _ = calculate_metrics(y_true, y_pred_list, labels=labels)
    
    # Create output dictionary with parameters and results
    output_data = {
        "params": vars(args),
        "results": metrics
    }
    
    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    
if __name__ == "__main__":
    main()
