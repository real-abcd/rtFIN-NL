#!/usr/bin/env bash
set -euo pipefail

# Run from this directory after creating env.sh.

# H100 x2 POC.
MODEL_SUITE=both \
BGE_EPOCHS=1 \
QWEN_MAX_STEPS=500 \
QWEN_MAX_SEQ_LENGTH=256 \
sbatch --partition=h100 --gres=gpu:2 job.sh

# H100 x4 scale check.
MODEL_SUITE=both \
BGE_EPOCHS=2 \
QWEN_MAX_STEPS=1500 \
QWEN_MAX_SEQ_LENGTH=512 \
sbatch --partition=h100 --gres=gpu:4 job.sh

# H100 x8 full candidate.
MODEL_SUITE=both \
BGE_EPOCHS=3 \
QWEN_MAX_STEPS=-1 \
QWEN_EPOCHS=1 \
QWEN_MAX_SEQ_LENGTH=512 \
sbatch --partition=h100 --gres=gpu:8 job.sh
