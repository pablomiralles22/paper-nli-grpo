import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from torchmetrics import Accuracy, F1Score, MetricCollection
from src.data_utils.multitask_learning.collator_fn import MultitaskCollatorFn
from src.data_utils.multitask_learning.preprocess_fn import preprocess_fn as aux_preprocess_fn
from src.data_utils.encoder.preprocess_fn import preprocess_fn as encoder_preprocess_fn
from src.freeze_layers import freeze_layers

os.environ["WANDB_PROJECT"] = "nli"  # name your W&B project

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a model for NLI classification on SNLI.')
    parser.add_argument('--model_name', type=str, required=True, help='Pre-trained model name from Hugging Face.')
    parser.add_argument('--run_name', type=str, required=True, help='Name of the run for logging.')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file for training.')
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def joint_preprocess_fn(examples, tokenizer, max_length, add_aux=True):
    classification_batch = {
        f"classification/{k}": v
        for k, v in encoder_preprocess_fn(examples, tokenizer, max_length).items()
    }
    if add_aux is False:
        return classification_batch

    aux_batch = {
        f"aux/{k}": v
        for k, v in aux_preprocess_fn(examples, tokenizer, max_length).items()
    }
    return {**classification_batch, **aux_batch}

class NLIDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for NLI datasets."""

    def __init__(
        self,
        data_path: str,
        model_name: str,
        max_length: int = 768,
        batch_size: int = 32,
        mlm_probability: float = 0.3,
    ):
        super().__init__()
        self.data_path = data_path
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.train_collator = MultitaskCollatorFn(self.tokenizer, mlm_probability)
        self.test_collator = MultitaskCollatorFn(self.tokenizer, apply_aux=False)

    def prepare_data(self):
        load_dataset(self.data_path)
        load_dataset('anli')

    def setup(self, stage: Optional[str] = None):
        anli_dataset = load_dataset('anli', split="test_r3")
        main_dataset = load_dataset(self.data_path, split="train")

        # Process the dataset
        tokenized_main_dataset = main_dataset.map(
            lambda examples: joint_preprocess_fn(examples, self.tokenizer, self.max_length),
            batched=True,
            remove_columns=main_dataset.column_names  # Remove all original columns
        )
        tokenized_anli_dataset = anli_dataset.map(
            lambda examples: joint_preprocess_fn(examples, self.tokenizer, self.max_length, add_aux=False),
            batched=True,
            remove_columns=anli_dataset.column_names  # Remove all original columns
        )

        trainval_dataset = tokenized_main_dataset.train_test_split(test_size=0.2)
        self.train_dataset = trainval_dataset["train"]
        # self.val_dataset = trainval_dataset["test"]
        self.val_dataset = tokenized_anli_dataset
        self.test_dataset = tokenized_anli_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=self.train_collator,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.test_collator,
            num_workers=8,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.test_collator,
            num_workers=8,
        )

class NLIClassifier(pl.LightningModule):
    """PyTorch Lightning module for NLI classification."""
    
    def __init__(self, model_name: str, optimizer_config: dict, num_frozen_layers: int = 0):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=3,  # NLI has 3 labels: entailment, contradiction, neutral
        )
        freeze_layers(self.model.base_model, num_frozen_layers)
        self.mlm_head = nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size)
        
        self.optimizer_config = optimizer_config

        self.train_metrics = MetricCollection({
            "accuracy": Accuracy(task="multiclass", num_classes=3),
            "f1": F1Score(task="multiclass", num_classes=3),
        }, prefix="train/")
        self.val_metrics = MetricCollection({
            "accuracy": Accuracy(task="multiclass", num_classes=3),
            "f1": F1Score(task="multiclass", num_classes=3),
        }, prefix="val/")
        self.test_metrics = MetricCollection({
            "accuracy": Accuracy(task="multiclass", num_classes=3),
            "f1": F1Score(task="multiclass", num_classes=3),
        }, prefix="test/")
        
        
    def forward(self, **inputs):
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, self.train_metrics, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, self.val_metrics, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, self.test_metrics, "test")

    def _step(self, batch, batch_idx, metrics_fn, prefix):
        outputs = self.model(**batch["classification"], output_hidden_states=True)

        classification_loss = outputs.loss
        
        metrics_fn(outputs.logits, batch["classification"]["labels"])

        self.log(f"{prefix}/classification_loss", classification_loss, prog_bar=True, logger=True)
        self.log_dict(metrics_fn, prog_bar=True, logger=True, on_step=(prefix=="train"), on_epoch=True)

        if "aux" not in batch:
            return classification_loss

        inputs = {k: v for k, v in batch["aux"].items() if k != "labels"}
        outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs["hidden_states"][-1]
        mlm_logits = self.mlm_head(last_hidden_state)

        # print(mlm_logits.shape)
        # print(batch["aux"]["labels"].shape)

        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, mlm_logits.size(-1)),
            batch["aux"]["labels"].view(-1),
            ignore_index=-100,
        )
        self.log(f"{prefix}/mlm_loss", mlm_loss, prog_bar=True, logger=True)

        return classification_loss + mlm_loss
    
    def configure_optimizers(self):
        scheduler_config = self.optimizer_config.pop('scheduler', None)

        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_config)
        scheduler = get_scheduler(optimizer=optimizer, **scheduler_config)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/classification_loss",
                "interval": "step",
            },
        }

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)

    data_module_config = config.get('data_module', dict())
    optimizer_config = config.get('optimizer', dict())
    model_config = config.get('model', dict())
    trainer_config = config.get('trainer', dict())
    
    # Create data module
    data_module = NLIDataModule(
        model_name=args.model_name,
        **data_module_config,
    )
    
    # Create model
    model = NLIClassifier(
        model_name=args.model_name,
        optimizer_config=optimizer_config,
        **model_config,
    )
    
    # Setup logger
    wandb_logger = WandbLogger(project="nli", name=args.run_name)
    
    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.get('output_dir', './out/nli_model'),
        filename='{epoch}-{val_f1:.2f}',
        save_top_k=1,
        monitor='val/f1'
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        **trainer_config,
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # # Save the model and tokenizer
    # model_save_path = config.get('output_dir', './out/nli_model')
    # model.model.save_pretrained(model_save_path)
    # data_module.tokenizer.save_pretrained(model_save_path)
    # print(f"Model saved to {model_save_path}")

    trainer.test(model, data_module, ckpt_path="best")

if __name__ == "__main__":
    main()
