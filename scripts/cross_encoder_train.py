import os
import sys
import hydra
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from src.csv_logger import CSVLogger

os.environ["WANDB_PROJECT"] = "nli"
os.environ["WANDB_LOG_MODEL"] = "end"

ENV_CONFIG = dotenv_values(".env")
HF_ACCESS_TOKEN = ENV_CONFIG.get("HF_WRITE_TOKEN")


def load_model_and_tokenizer(model_id, num_labels=3, model_init_kwargs=None):
    model_init_kwargs = model_init_kwargs or {}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        **model_init_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_ACCESS_TOKEN)

    # Decoder-only tokenizers often do not define a pad token.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                f"Tokenizer for '{model_id}' has no pad_token/eos_token. "
                "Set a pad token before training."
            )
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id

    if "attn_implementation" in model_init_kwargs and model_init_kwargs["attn_implementation"] == "flash_attention_2":
        tokenizer.padding_side = "left"

    return model, tokenizer


def load_data(
    tokenizer,
    data_path,
    split,
    seed=42,
    data_name=None,
    premise_column="premise",
    hypothesis_column="hypothesis",
    label_column="label",
    max_length=1024,
):
    dataset = load_dataset(data_path, name=data_name, split=split, token=HF_ACCESS_TOKEN)
    dataset = dataset.filter(lambda item: item[label_column] != -1)
    dataset = dataset.shuffle(seed=seed)

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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


@hydra.main(version_base=None, config_path="../configs/cross_encoder", config_name="base")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_object(cfg)

    model_id_safe = cfg["model_id"].replace("/", "-")
    run_name = cfg.get("run_name", model_id_safe + "-cross-encoder")
    output_dir = os.path.join(cfg.get("output_dir", "out/cross_encoder_train"), run_name)
    seed = cfg.get("seed", 42)

    trainer_config = cfg["trainer_config"]
    model_init_kwargs = trainer_config.pop("model_init_kwargs", {})
    if HF_ACCESS_TOKEN is not None:
        model_init_kwargs["token"] = HF_ACCESS_TOKEN

    model, tokenizer = load_model_and_tokenizer(
        cfg["model_id"],
        num_labels=cfg.get("num_labels", 3),
        model_init_kwargs=model_init_kwargs,
    )

    if "lora_config" in cfg:
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, **cfg["lora_config"])
        is_quantized = any(method in cfg["model_id"].lower() for method in ["awq", "gptq"])
        if is_quantized:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=trainer_config.get("gradient_checkpointing", False),
            )
        elif trainer_config.get("gradient_checkpointing", False):
            model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)

    train_dataset = load_data(
        tokenizer=tokenizer,
        data_path=cfg["data_path"],
        data_name=cfg.get("data_name"),
        split=cfg.get("train_split", cfg.get("data_split", "train")),
        seed=seed,
        premise_column=cfg.get("premise_column", "premise"),
        hypothesis_column=cfg.get("hypothesis_column", "hypothesis"),
        label_column=cfg.get("label_column", "label"),
        max_length=cfg.get("max_length", 1024),
    )

    eval_dataset = None
    eval_split = cfg.get("eval_split")
    if eval_split is not None:
        eval_dataset = load_data(
            tokenizer=tokenizer,
            data_path=cfg.get("eval_data_path", cfg["data_path"]),
            data_name=cfg.get("eval_data_name", cfg.get("data_name")),
            split=eval_split,
            seed=seed,
            premise_column=cfg.get("premise_column", "premise"),
            hypothesis_column=cfg.get("hypothesis_column", "hypothesis"),
            label_column=cfg.get("label_column", "label"),
            max_length=cfg.get("max_length", 1024),
        )

    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=output_dir,
        **trainer_config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
        callbacks=[CSVLogger(output_dir=output_dir)],
    )

    resume_from_checkpoint = True if cfg.get("use_checkpoint", False) else None
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
