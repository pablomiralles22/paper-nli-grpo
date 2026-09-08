import os
import subprocess

from itertools import product
from tqdm import tqdm

SEEDS = [0, 22, 42]

BASE_PARAMS = {
    "batch_size": 16,
    "prompt_config": "configs/prompts/deepseek.yaml",
    "max_lora_rank": 64,
    "binary_mode": False,
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

TEMPERATURES = [
    {"temperature": 0.0, "num_generations": 1},
    # {"temperature": 0.5, "num_generations": 5},
    # {"temperature": 0.9, "num_generations": 5},
    # {"temperature": 1.2, "num_generations": 5},
]

MODELS = [
    # {"base_model": "Qwen/Qwen2.5-7B-Instruct", "quantization": "None", "adapter_path": "None", "run_name": "Qwen2.5-7B", "gpus": 1},
    # {"base_model": "Qwen/Qwen2.5-7B-Instruct-AWQ", "quantization": "awq", "adapter_path": "None", "run_name": "Qwen2.5-7B_AWQ", "gpus": 1},
    # {"base_model": "Qwen/Qwen2.5-14B-Instruct-AWQ", "quantization": "awq", "adapter_path": "None", "run_name": "Qwen2.5-14B_AWQ", "gpus": 2},
    # {"base_model": "Qwen/Qwen2.5-32B-Instruct-AWQ", "quantization": "awq", "adapter_path": "None", "run_name": "Qwen2.5-32B_AWQ", "gpus": 2},

    # Main GRPO models
    {"base_model": "Qwen/Qwen2.5-7B-Instruct-AWQ", "quantization": "awq",
     "adapter_path": "out/rl_train/seed_{seed}/Qwen-7B_AWQ-r64/checkpoint-4000", "run_name": "Qwen2.5-7B_AWQ-GRPO-r64", "max_lora_rank": 64, "gpus": 1},
    {"base_model": "Qwen/Qwen2.5-14B-Instruct-AWQ", "quantization": "awq",
     "adapter_path": "out/rl_train/seed_{seed}/Qwen-14B_AWQ-r64/checkpoint-4000", "run_name": "Qwen2.5-14B_AWQ-GRPO-r64", "max_lora_rank": 64, "gpus": 2},
    {"base_model": "Qwen/Qwen2.5-32B-Instruct-AWQ", "quantization": "awq",
     "adapter_path": "out/rl_train/seed_{seed}/Qwen-32B_AWQ-r64/checkpoint-4000", "run_name": "Qwen2.5-32B_AWQ-GRPO-r64", "max_lora_rank": 64, "gpus": 2},

    # LoRA rank ablation
    {"base_model": "Qwen/Qwen2.5-7B-Instruct", "quantization": "None",
     "adapter_path": "out/rl_train/seed_{seed}/Qwen-7B-r8/checkpoint-1000", "run_name": "Qwen2.5-7B-GRPO-r8", "max_lora_rank": 8, "gpus": 1},
    {"base_model": "Qwen/Qwen2.5-7B-Instruct", "quantization": "None",
     "adapter_path": "out/rl_train/seed_{seed}/Qwen-7B-r16/checkpoint-1000", "run_name": "Qwen2.5-7B-GRPO-r16", "max_lora_rank": 16, "gpus": 1},
    {"base_model": "Qwen/Qwen2.5-7B-Instruct", "quantization": "None",
     "adapter_path": "out/rl_train/seed_{seed}/Qwen-7B-r32/checkpoint-1000", "run_name": "Qwen2.5-7B-GRPO-r32", "max_lora_rank": 32, "gpus": 1},
    {"base_model": "Qwen/Qwen2.5-7B-Instruct", "quantization": "None",
     "adapter_path": "out/rl_train/seed_{seed}/Qwen-7B-r64/checkpoint-1000", "run_name": "Qwen2.5-7B-GRPO-r64", "max_lora_rank": 64, "gpus": 1},
    {"base_model": "Qwen/Qwen2.5-7B-Instruct", "quantization": "None",
     "adapter_path": "out/rl_train/seed_{seed}/Qwen-7B-r128/checkpoint-1000", "run_name": "Qwen2.5-7B-GRPO-r128", "max_lora_rank": 128, "gpus": 1},

    {"base_model": "Qwen/Qwen2.5-7B-Instruct-AWQ", "quantization": "awq",
     "adapter_path": "out/rl_train/seed_{seed}/Qwen-7B_AWQ-r8/checkpoint-1000", "run_name": "Qwen2.5-7B_AWQ-GRPO-r8", "max_lora_rank": 8, "gpus": 1},
    {"base_model": "Qwen/Qwen2.5-7B-Instruct-AWQ", "quantization": "awq",
     "adapter_path": "out/rl_train/seed_{seed}/Qwen-7B_AWQ-r16/checkpoint-1000", "run_name": "Qwen2.5-7B_AWQ-GRPO-r16", "max_lora_rank": 16, "gpus": 1},
    {"base_model": "Qwen/Qwen2.5-7B-Instruct-AWQ", "quantization": "awq",
     "adapter_path": "out/rl_train/seed_{seed}/Qwen-7B_AWQ-r32/checkpoint-1000", "run_name": "Qwen2.5-7B_AWQ-GRPO-r32", "max_lora_rank": 32, "gpus": 1},
    {"base_model": "Qwen/Qwen2.5-7B-Instruct-AWQ", "quantization": "awq",
     "adapter_path": "out/rl_train/seed_{seed}/Qwen-7B_AWQ-r64_bs32/checkpoint-1000", "run_name": "Qwen2.5-7B_AWQ-GRPO-r64_bs32", "max_lora_rank": 64, "gpus": 1},
    {"base_model": "Qwen/Qwen2.5-7B-Instruct-AWQ", "quantization": "awq",
     "adapter_path": "out/rl_train/seed_{seed}/Qwen-7B_AWQ-r128/checkpoint-1000", "run_name": "Qwen2.5-7B_AWQ-GRPO-r128", "max_lora_rank": 128, "gpus": 1},
]

COMMAND_TEMPLATE = """
gbatch --gpus {gpus} -- \
VLLM_USE_V1=0 uv run python3 scripts/gen_model_eval.py \
    --run-name {run_name} \
    --model-id {base_model} --quantization {quantization} --lora-adapter-path {adapter_path} --max-lora-rank {max_lora_rank} \
    --batch-size {batch_size} \
    --data-path {data_path} --data-name {data_name} --split {data_split} --binary-mode {binary_mode} \
    --prompt-config {prompt_config} \
    --output-file {output_file} \
    --temperature {temperature} \
    --num-generations {num_generations}
""".strip()

def main():
    for seed in SEEDS:
        for model, temperature, dataset in tqdm(product(MODELS, TEMPERATURES, TEST_DATASETS)):
            data_path_safe = dataset["data_path"].replace("/", "-")
            data_name_str = f"_{dataset['data_name']}" if dataset["data_name"] is not None else ""
            adapter_path = model["adapter_path"].format(seed=seed) if model["adapter_path"] != "None" else "None"

            output_file = f"out/eval/grpo/{model['run_name']}_seed={seed}_{data_path_safe}{data_name_str}_{dataset['data_split']}_{temperature['temperature']}.json"
            if os.path.exists(output_file):
                print(f"Skipping {output_file} as it already exists.")
                continue
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            params = {
                **BASE_PARAMS,
                **model,
                **dataset,
                **temperature,
                "output_file": output_file,
                "adapter_path": adapter_path,
            }
            command = COMMAND_TEMPLATE.format(**params)
            print(command)
            subprocess.run(command, shell=True)

if __name__ == "__main__":
    main()
