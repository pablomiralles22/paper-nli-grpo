import os
import subprocess

from itertools import product
from tqdm import tqdm

SEEDS = [0, 22, 42]

TARGET_MODULES = "[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]"

# Mirrors launchers/grpo_train.py experiment definitions.
EXPERIMENTS = [
    # Main  models
    {
        "run_name": "Qwen-7B_AWQ-r64",
        "model_id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "gpus": 1,
        "overrides": [
            "trainer_config.per_device_train_batch_size=32",
            "trainer_config.gradient_accumulation_steps=1",
            "trainer_config.per_device_eval_batch_size=32",
            "++trainer_config.gradient_checkpointing=true",
            f"lora_config.target_modules=\"{TARGET_MODULES}\"",
        ],
    },
    {
        "run_name": "Qwen-7B-r64",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "gpus": 1,
        "overrides": [
            "trainer_config.per_device_train_batch_size=8",
            "trainer_config.gradient_accumulation_steps=4",
            "trainer_config.per_device_eval_batch_size=8",
            "trainer_config.bf16=true",
            "trainer_config.fp16=false",
            "trainer_config.model_init_kwargs.torch_dtype=bfloat16",
            "++trainer_config.gradient_checkpointing=true",
        ],
    },
    {
        "run_name": "Qwen-14B_AWQ-r64",
        "model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "gpus": 1,
        "overrides": [
            "trainer_config.per_device_train_batch_size=8",
            "trainer_config.gradient_accumulation_steps=4",
            "trainer_config.per_device_eval_batch_size=8",
            "++trainer_config.gradient_checkpointing=true",
            f"lora_config.target_modules=\"{TARGET_MODULES}\"",
        ],
    },
    {
        "run_name": "Qwen-32B_AWQ-r64",
        "model_id": "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "gpus": 1,
        "overrides": [
            "trainer_config.per_device_train_batch_size=4",
            "trainer_config.gradient_accumulation_steps=8",
            "trainer_config.per_device_eval_batch_size=4",
            "++trainer_config.gradient_checkpointing=true",
            f"lora_config.target_modules=\"{TARGET_MODULES}\"",
        ],
    },
]

def build_command(seed: int, exp: dict) -> str:
    output_dir = f"out/train/cross_encoder/seed_{seed}"
    override_str = " \\\n    ".join(exp["overrides"])

    return (
        f"gbatch --gpus {exp['gpus']} -- \\\n"
        f"uv run python3 scripts/cross_encoder_train.py \\\n"
        f"    run_name={exp['run_name']} model_id={exp['model_id']} \\\n"
        f"    output_dir={output_dir} ++seed={seed} \\\n"
        f"    {override_str}"
    )


def main() -> None:
    total = len(SEEDS) * len(EXPERIMENTS)

    for seed, exp in tqdm(product(SEEDS, EXPERIMENTS), total=total):
        cmd = build_command(seed, exp)
        print(cmd)
        subprocess.run(cmd, shell=True, check=False)


if __name__ == "__main__":
    main()
