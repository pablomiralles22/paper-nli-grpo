import os
import subprocess

from itertools import product
from tqdm import tqdm

SEEDS = [22, 42]

BASE_PARAMS = {
    "batch_size": 16,
    "dtype": "auto",
    "attn_implementation": "flash_attention_2",
}

TEST_DATASETS = [
    {"data_path": "anli", "data_split": "test_r1", "data_name": None},
    {"data_path": "anli", "data_split": "test_r2", "data_name": None},
    {"data_path": "anli", "data_split": "test_r3", "data_name": None},
    {"data_path": "pablomiralles22/nli-diagnostic", "data_split": "test_know", "data_name": None},
    {"data_path": "pablomiralles22/nli-diagnostic", "data_split": "test_logic", "data_name": None},
    {"data_path": "pablomiralles22/nli-diagnostic", "data_split": "test_ls", "data_name": None},
    {"data_path": "pablomiralles22/nli-diagnostic", "data_split": "test_pas", "data_name": None},
    {"data_path": "snli", "data_split": "test", "data_name": None},
    {"data_path": "nyu-mll/glue", "data_split": "validation_mismatched", "data_name": "mnli"},
    {"data_path": "pablomiralles22/hans", "data_split": "test_lex", "data_name": None, "binary_mode": True},
    {"data_path": "pablomiralles22/hans", "data_split": "test_sub", "data_name": None, "binary_mode": True},
    {"data_path": "pablomiralles22/hans", "data_split": "test_cons", "data_name": None, "binary_mode": True},
    {"data_path": "pablomiralles22/counter-nli", "data_split": "test", "data_name": None},
]

# Mirrors launchers/cross_encoder_train.py experiment definitions.
EXPERIMENTS = [
    {
        "run_name": "Qwen-7B_AWQ-r64",
        "model_id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "gpus": 1,
    },
    {
        "run_name": "Qwen-7B-r64",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "gpus": 1,
    },
    {
        "run_name": "Qwen-14B_AWQ-r64",
        "model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "gpus": 1,
    },
    {
        "run_name": "Qwen-32B_AWQ-r64",
        "model_id": "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "gpus": 1,
    },
]

COMMAND_TEMPLATE = """
gbatch --gpus {gpus} -- \\
uv run python3 scripts/cross_encoder_eval.py \\
    --run-name {run_name} \\
    --model-id {model_id} \\
    --lora-adapter-path {lora_adapter_path} \\
    --data-path {data_path} --data-name {data_name} --split {data_split} \\
    --batch-size {batch_size} \\
    --binary-mode {binary_mode} \\
    --dtype {dtype} \\
    --attn-implementation {attn_implementation} \\
    --output-file {output_file}
""".strip()


def main() -> None:
    total = len(SEEDS) * len(EXPERIMENTS) * len(TEST_DATASETS)

    for seed, exp, dataset in tqdm(product(SEEDS, EXPERIMENTS, TEST_DATASETS), total=total):
        data_path_safe = dataset["data_path"].replace("/", "-")
        data_name_str = f"_{dataset['data_name']}" if dataset["data_name"] is not None else ""

        output_file = (
            f"out/eval/cross_encoder/{exp['run_name']}_seed={seed}_"
            f"{data_path_safe}{data_name_str}_{dataset['data_split']}.json"
        )
        if os.path.exists(output_file):
            print(f"Skipping {output_file} as it already exists.")
            continue

        lora_adapter_path = f"out/train/cross_encoder/seed_{seed}/{exp['run_name']}"
        params = {
            **BASE_PARAMS,
            **exp,
            **dataset,
            "binary_mode": dataset.get("binary_mode", False),
            "lora_adapter_path": lora_adapter_path,
            "output_file": output_file,
        }

        command = COMMAND_TEMPLATE.format(**params)
        print(command)
        subprocess.run(command, shell=True, check=False)


if __name__ == "__main__":
    main()
