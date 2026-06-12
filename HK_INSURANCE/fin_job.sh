#!/bin/bash
#SBATCH --job-name=hkfire_serve
#SBATCH --gres-flags=enforce-binding
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=hkfire_serve_%j.out
#SBATCH --error=hkfire_serve_%j.err

set -euo pipefail

BASE_DIR=/mnt/cepheid/projects/korail
HF_CACHE=$BASE_DIR/hf_cache
SIF_PATH=$BASE_DIR/vllm.sif

export HF_HOME="$HF_CACHE"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

echo "Starting Qwen3.5 vLLM servers on H100"
echo "BASE_DIR=$BASE_DIR"
echo "SIF_PATH=$SIF_PATH"

# 9B fallback server
# - 중간 길이 입력까지 감안
# - 동시성은 너무 높이지 않고 안정성 우선
CUDA_VISIBLE_DEVICES=1 singularity exec --nv \
    --bind "$BASE_DIR:$BASE_DIR" \
    "$SIF_PATH" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model cyankiwi/Qwen3.5-9B-AWQ-4bit \
        --port 11052 \
        --served-model-name cyankiwi/Qwen3.5-9B-AWQ-4bit \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.92 \
        --max-model-len 32768 \
        --max-num-seqs 32 \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --attention-backend FLASHINFER \
        --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":1}' \
        --trust-remote-code &

PID_9B=$!

# 27B primary server
# - 긴 분석 입력, 큰 표/엑셀 값, 긴 문맥을 우선 처리
# - H100이면 여기서 넉넉하게 가져가는 편이 맞음
CUDA_VISIBLE_DEVICES=0 singularity exec --nv \
    --bind "$BASE_DIR:$BASE_DIR" \
    "$SIF_PATH" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model cyankiwi/Qwen3.5-27B-AWQ-4bit \
        --port 11051 \
        --served-model-name cyankiwi/Qwen3.5-27B-AWQ-4bit \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.92 \
        --max-model-len 65536 \
        --max-num-seqs 16 \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --attention-backend FLASHINFER \
        --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":1}' \
        --trust-remote-code &

PID_27B=$!

trap 'echo "Stopping servers..."; kill $PID_9B $PID_27B 2>/dev/null || true' INT TERM

wait
