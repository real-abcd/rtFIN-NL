# H100 Slurm Embedding Runs

This folder is the Slurm entrypoint for scaling the A100 embedding-tuning flow onto H100 nodes.

Canonical entrypoint:

- `job.sh`: submit this file with `sbatch job.sh`

Legacy alias:

- `sbatch_h100_train.sh`: same job logic, kept for compatibility

Fixed models:

- BGE: `dragonkue/BGE-m3-ko`
- Qwen: `Qwen/Qwen3-Embedding-4B`

## Data source

The job downloads the training inputs from Hugging Face automatically:

- `hyunseop/Nemotron-Personas-Korea-Embedding-ContrastiveLearning`
- `train_bge_m3.jsonl`
- `train_qwen3.jsonl`

The raw files are cached under `./source_data/` by default.

## What the job does

The batch script boots a local venv on the H100 node, installs the project dependencies if they are present, downloads the dataset files, splits the train JSONL files into train/eval slices, runs the selected model suite, and writes a `summary.json` at the end.

Outputs land under:

```bash
./output/<run_id>/
```

Logs are tee'd to:

```bash
./logs/job_<slurm_job_id>.log
```

## Setup

```bash
cd /mnt/cepheid/users/hsypfsv/rtFIN_ISL_ML/embedding_tuning/h100_slurm
cp env.example env.sh
vi env.sh
```

## Submit

```bash
sbatch --partition=h100 --gres=gpu:2 job.sh
```

## Common profiles

### H100 x2 POC

```bash
MODEL_SUITE=both \
BGE_EPOCHS=1 \
QWEN_MAX_STEPS=500 \
QWEN_MAX_SEQ_LENGTH=256 \
sbatch --partition=h100 --gres=gpu:2 job.sh
```

### H100 x4 scale check

```bash
MODEL_SUITE=both \
BGE_EPOCHS=2 \
QWEN_MAX_STEPS=1500 \
QWEN_MAX_SEQ_LENGTH=512 \
sbatch --partition=h100 --gres=gpu:4 job.sh
```

### H100 x8 full candidate

```bash
MODEL_SUITE=both \
BGE_EPOCHS=3 \
QWEN_MAX_STEPS=-1 \
QWEN_EPOCHS=1 \
QWEN_MAX_SEQ_LENGTH=512 \
sbatch --partition=h100 --gres=gpu:8 job.sh
```

## Notes

- Keep Qwen at `batch/GPU=1` until the H100 memory profile is stable.
- `INSTALL_AXOLOTL=1` is available if your project stack needs it, but it stays off by default.
- `submit_examples.sh` contains the same launch examples in executable form.
