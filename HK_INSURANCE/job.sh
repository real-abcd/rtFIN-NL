#!/bin/bash
#SBATCH --job-name=hkfire_serve
#SBATCH --gres-flags=enforce-binding
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2          # H100 총 2장 요청
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

BASE_DIR=/mnt/cepheid/projects/korail
HF_CACHE=$BASE_DIR/hf_cache
SIF_PATH=$BASE_DIR/vllm.sif

export HF_HOME=$HF_CACHE
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
echo "Starting Qwen3.5 BF16 Servers on Separate GPUs"

# ====== 1번 서버: Qwen 27B급 (GPU 0번 단독 사용) ======
# Fixed by adding --max-num-seqs 800 to match available Mamba/KV cache blocks
CUDA_VISIBLE_DEVICES=0 singularity exec --nv \
    --bind $BASE_DIR:$BASE_DIR \
    $SIF_PATH \
    python3 -m vllm.entrypoints.openai.api_server \
        --model cyankiwi/Qwen3.5-27B-AWQ-4bit \
        --port 11051 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 32000 \
        --max-num-seqs 800 \
        --trust-remote-code &

# ====== 2번 서버: Qwen 9B급 (GPU 1번 단독 사용) ======
# This one has a shorter context (8192) and smaller size, so it likely runs fine as-is.
CUDA_VISIBLE_DEVICES=1 singularity exec --nv \
    --bind $BASE_DIR:$BASE_DIR \
    $SIF_PATH \
    python3 -m vllm.entrypoints.openai.api_server \
        --model cyankiwi/Qwen3.5-9B-AWQ-4bit \
        --port 11052 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8192 \
        --trust-remote-code &

# 두 서버가 배경에서 죽지 않고 계속 대기하도록 설정
wait
