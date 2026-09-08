# NLI

Code for paper "Evaluating the Scalability and Adversarial Generalization of GRPO-Trained NLI Models" (arxiv 2504.18376).

## 1. Datasets

* **SNLI**: `snli`
* **MNLI**: `nyu-mll/glue`, subset `mnli`
* **ANLI**: `anli`
* **Diagnostics**: `pablomiralles22/nli-diagnostic`
* **HANS**: `pablomiralles22/hans`
* **Counter-SNLI**: `tasksource/counterfactually-augmented-snli` -> `pablomiralles22/counter-nli`

The download process and/or applied transformations for the last three datasets is found in `notebooks/datasets.ipynb`.

## Experiments

### Main experiments

The main experiments are launched from the scripts in the `launchers` directory, using [`gflow`](https://www.runqd.com/) to handle GPU dispatching and `uv` as package manager.

### Encoder-decoder replication of "Prompting for explanations improves Adversarial NLI. Is this true? {Yes} it is {true} because {it weakens superficial cues}"

**Training**

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python3 scripts/encoder_decoder/train.py \
    --input_model_id google-t5/t5-3b \
    --run_name encoder_decoder/t5-3b \
    --trainer_config configs/encoder_decoder/base.yaml
```

**Testing**
```bash
CKPT_PATH="out/encoder_decoder/t5-3b-v2/checkpoint-38000"
CONDA_EXE="/home/pablo/miniconda3/bin/conda"

# ANLI
splits=("r1" "r2" "r3")
for split in "${splits[@]}"; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 TOKENIZERS_PARALLELISM=false \
  $CONDA_EXE run -n nli_v3 python3 scripts/encoder_decoder/train.py \
    --input_model $CKPT_PATH \
    --run_name encoder_decoder/t5-3b-anli-test_${split} \
    --trainer_config configs/encoder_decoder/base.yaml \
    --test_only \
    --data_path anli \
    --train_split test_$split --val_split test_$split --test_split test_$split
done

# NLI Diagnostic
splits=("know" "logic" "ls" "pas")
for split in "${splits[@]}"; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 TOKENIZERS_PARALLELISM=false \
  $CONDA_EXE run -n nli_v3 python3 scripts/encoder_decoder/train.py \
    --input_model $CKPT_PATH \
    --run_name encoder_decoder/t5-3b-nli-diag-test_${split} \
    --trainer_config configs/encoder_decoder/base.yaml \
    --test_only \
    --data_path pablomiralles22/nli-diagnostic \
    --train_split test_$split --val_split test_$split --test_split test_$split
done

# HANS
splits=("lex" "sub" "cons")
for split in "${splits[@]}"; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 TOKENIZERS_PARALLELISM=false \
  $CONDA_EXE run -n nli_v3 python3 scripts/encoder_decoder/train.py \
    --input_model $CKPT_PATH \
    --run_name encoder_decoder/t5-3b-hans-test_${split} \
    --trainer_config configs/encoder_decoder/base.yaml \
    --test_only --binary_mode \
    --data_path pablomiralles22/hans \
    --train_split test_$split --val_split test_$split --test_split test_$split
done

# NLI Counter
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 TOKENIZERS_PARALLELISM=false \
 $CONDA_EXE run -n nli_v3 python3 scripts/encoder_decoder/train.py \
    --input_model $CKPT_PATH \
    --run_name encoder_decoder/t5-3b-counter_nli-test \
    --trainer_config configs/encoder_decoder/base.yaml \
    --test_only \
    --data_path pablomiralles22/counter-nli \
    --train_split test --val_split test --test_split test

# SNLI
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 TOKENIZERS_PARALLELISM=false \
 $CONDA_EXE run -n nli_v3 python3 scripts/encoder_decoder/train.py \
    --input_model $CKPT_PATH \
    --run_name encoder_decoder/t5-3b-snli-test \
    --trainer_config configs/encoder_decoder/base.yaml \
    --test_only \
    --data_path snli \
    --train_split test --val_split test --test_split test
```