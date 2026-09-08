import os
import pandas as pd
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

class CSVLogger(TrainerCallback):
    """
    A simple CSV logger callback for Hugging Face Trainer.
    Logs metrics to a CSV file in the specified output directory.
    """
    def __init__(self, output_dir: str, filename: str = "metrics.csv"):
        self.output_dir = output_dir
        self.output_path = os.path.join(output_dir, filename)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return

        # Prepare metrics, filtering for scalars
        metrics = {"step": state.global_step, "epoch": state.epoch}
        for k, v in logs.items():
            if isinstance(v, (int, float, str)):
                metrics[k] = v

        os.makedirs(self.output_dir, exist_ok=True)

        if not os.path.exists(self.output_path):
            df = pd.DataFrame([metrics])
            df.to_csv(self.output_path, index=False)
        else:
            try:
                # Load existing data to handle potential new columns gracefully
                df = pd.read_csv(self.output_path)
                new_row = pd.DataFrame([metrics])
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(self.output_path, index=False)
            except Exception as e:
                # Fallback to simple append if pandas fails or file is corrupted
                import csv
                # We try to use the keys from the current metrics
                file_exists = os.path.exists(self.output_path)
                with open(self.output_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(metrics.keys()), extrasaction="ignore")
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(metrics)
