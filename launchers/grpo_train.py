import os
import subprocess

from itertools import product
from tqdm import tqdm

SEEDS = [0, 22, 42]

TARGET_MODULES = "[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]"

# Experiments from scripts/scratch/temp_launch_seed.sh, expanded to all eval seeds.
EXPERIMENTS = [
    # Main AWQ models (longer run)
    {
        "run_name": "Qwen-7B_AWQ-r64",
        "model_id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "gpus": 2,
        "overrides": [
            "trainer_config.per_device_train_batch_size=8",
            "trainer_config.per_device_eval_batch_size=8",
            "trainer_config.max_steps=4000",
            f"lora_config.target_modules=\"{TARGET_MODULES}\"",
        ],
        "expected_checkpoint": 4000,
    },
    {
        "run_name": "Qwen-14B_AWQ-r64",
        "model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "gpus": 2,
        "overrides": [
            "trainer_config.per_device_train_batch_size=8",
            "trainer_config.per_device_eval_batch_size=8",
            "trainer_config.max_steps=4000",
            f"lora_config.target_modules=\"{TARGET_MODULES}\"",
        ],
        "expected_checkpoint": 4000,
    },
    {
        "run_name": "Qwen-32B_AWQ-r64",
        "model_id": "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "gpus": 3,
        "overrides": [
            "trainer_config.per_device_train_batch_size=8",
            "trainer_config.per_device_eval_batch_size=8",
            "trainer_config.max_steps=4000",
            f"lora_config.target_modules=\"{TARGET_MODULES}\"",
        ],
        "expected_checkpoint": 4000,
    },

    # 7B LoRA rank ablations (base defaults for batch/max_steps)
    {
        "run_name": "Qwen-7B-r8",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "gpus": 1,
        "overrides": [
            "lora_config.r=8",
            "lora_config.lora_alpha=8",
            f"lora_config.target_modules=\"{TARGET_MODULES}\"",
        ],
        "expected_checkpoint": 1000,
    },
    {
        "run_name": "Qwen-7B-r16",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "gpus": 1,
        "overrides": [
            "lora_config.r=16",
            "lora_config.lora_alpha=16",
            f"lora_config.target_modules=\"{TARGET_MODULES}\"",
        ],
        "expected_checkpoint": 1000,
    },
    {
        "run_name": "Qwen-7B-r32",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "gpus": 1,
        "overrides": [
            "lora_config.r=32",
            "lora_config.lora_alpha=32",
            f"lora_config.target_modules=\"{TARGET_MODULES}\"",
        ],
        "expected_checkpoint": 1000,
    },
    {
        "run_name": "Qwen-7B-r64",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "gpus": 1,
        "overrides": [
            "lora_config.r=64",
            "lora_config.lora_alpha=64",
            f"lora_config.target_modules=\"{TARGET_MODULES}\"",
        ],
        "expected_checkpoint": 1000,
    },
    {
        "run_name": "Qwen-7B-r128",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "gpus": 1,
        "overrides": [
            "lora_config.r=128",
            "lora_config.lora_alpha=128",
            f"lora_config.target_modules=\"{TARGET_MODULES}\"",
        ],
        "expected_checkpoint": 1000,
    },

    # 7B AWQ LoRA rank ablations
    {
        "run_name": "Qwen-7B_AWQ-r8",
        "model_id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "gpus": 1,
        "overrides": [
            "lora_config.r=8",
            "lora_config.lora_alpha=8",
            f"lora_config.target_modules=\"{TARGET_MODULES}\"",
        ],
        "expected_checkpoint": 1000,
    },
    {
        "run_name": "Qwen-7B_AWQ-r16",
        "model_id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "gpus": 1,
        "overrides": [
            "lora_config.r=16",
            "lora_config.lora_alpha=16",
            f"lora_config.target_modules=\"{TARGET_MODULES}\"",
        ],
        "expected_checkpoint": 1000,
    },
    {
        "run_name": "Qwen-7B_AWQ-r32",
        "model_id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "gpus": 1,
        "overrides": [
            "lora_config.r=32",
            "lora_config.lora_alpha=32",
            f"lora_config.target_modules=\"{TARGET_MODULES}\"",
        ],
        "expected_checkpoint": 1000,
    },
    {
        "run_name": "Qwen-7B_AWQ-r64_bs32",
        "model_id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "gpus": 1,
        "overrides": [
            "lora_config.r=64",
            "lora_config.lora_alpha=64",
            f"lora_config.target_modules=\"{TARGET_MODULES}\"",
        ],
        "expected_checkpoint": 1000,
    },
    {
        "run_name": "Qwen-7B_AWQ-r128",
        "model_id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "gpus": 1,
        "overrides": [
            "lora_config.r=128",
            "lora_config.lora_alpha=128",
            f"lora_config.target_modules=\"{TARGET_MODULES}\"",
        ],
        "expected_checkpoint": 1000,
    },
]


def checkpoint_exists(output_dir: str, run_name: str, checkpoint_step: int) -> bool:
    return os.path.exists(os.path.join(output_dir, run_name, f"checkpoint-{checkpoint_step}"))


def build_command(seed: int, exp: dict) -> str:
    output_dir = f"out/grpo_train/seed_{seed}"
    override_str = " \\\n    ".join(exp["overrides"])

    return (
        f"gbatch --gpus {exp['gpus']} -- \\\n"
        f"uv run python3 scripts/grpo_train.py \\\n"
        f"    run_name={exp['run_name']} model_id={exp['model_id']} \\\n"
        f"    output_dir={output_dir} ++seed={seed} \\\n"
        f"    {override_str}"
    )


def main() -> None:
    total = len(SEEDS) * len(EXPERIMENTS)

    for seed, exp in tqdm(product(SEEDS, EXPERIMENTS), total=total):
        output_dir = f"out/grpo_train/seed_{seed}"
        if checkpoint_exists(output_dir, exp["run_name"], exp["expected_checkpoint"]):
            print(
                f"Skipping {exp['run_name']} seed={seed}: "
                f"checkpoint-{exp['expected_checkpoint']} exists."
            )
            continue

        cmd = build_command(seed, exp)
        print(cmd)
        subprocess.run(cmd, shell=True, check=False)


if __name__ == "__main__":
    main()
