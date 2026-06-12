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
export HUGGING_FACE_HUB_TOKEN=$HF_HUB
echo "Starting Qwen3.5 Native MTP-Accelerated Servers"

# ====== 1번 서버: Qwen 27B급 (GPU 0번 단독 사용) ======
# - --max-num-seqs 800 으로 Mamba/KV 캐시 부족 에러 방지
# - --enable-prefix-caching 으로 공통 프롬프트 캐싱
# - qwen3_next_mtp 메소드로 내장 Multi-Token Prediction 가속 (1토큰 예측)
#   (*주의: MTP 활성화 시 대량 동시 요청보다는 지연시간(Latency) 단축에 최적화됩니다)# 

# ====== 1번 서버: Qwen 27B급 (GPU 0번 단독 사용) ======
# 에러 메시지에 맞춰 --max-num-seqs를 512(또는 764)로 하향 조정
CUDA_VISIBLE_DEVICES=0 singularity exec --nv \
    --bind $BASE_DIR:$BASE_DIR \
    $SIF_PATH \
    python3 -m vllm.entrypoints.openai.api_server \
        --model cyankiwi/Qwen3.5-27B-AWQ-4bit \
        --port 11051 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 32000 \
        --max-num-seqs 512 \
        --enable-prefix-caching \
        --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":1}' \
        --trust-remote-code &

# ====== 2번 서버: Qwen 9B급 (GPU 1번 단독 사용) ======
# - 동일하게 qwen3_next_mtp 및 추론 파서 셋업 적용
CUDA_VISIBLE_DEVICES=1 singularity exec --nv \
    --bind $BASE_DIR:$BASE_DIR \
    $SIF_PATH \
    python3 -m vllm.entrypoints.openai.api_server \
        --model cyankiwi/Qwen3.5-9B-AWQ-4bit \
        --port 11052 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8192 \
        --enable-prefix-caching \
        --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":1}' \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --trust-remote-code &

# 두 서버가 배경에서 죽지 않고 계속 대기하도록 설정
wait
