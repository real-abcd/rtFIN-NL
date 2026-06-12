#!/usr/bin/env bash
#SBATCH --job-name=rtfin_embed_h100
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=512GB
#SBATCH --time=72:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
export PYTHONNOUSERSITE=0
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="$(cd "$SUBMIT_DIR" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
RUN_ROOT="$SCRIPT_DIR/output"
mkdir -p "$LOG_DIR" "$RUN_ROOT"
LOG_FILE="$LOG_DIR/job_${SLURM_JOB_ID:-manual}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

if [ -f "$SCRIPT_DIR/env.sh" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/env.sh"
elif [ -f "$SCRIPT_DIR/env.example" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/env.example"
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATA_SOURCE_DIR="${DATA_SOURCE_DIR:-$SCRIPT_DIR/source_data}"
HF_DATASET_REPO="${HF_DATASET_REPO:-hyunseop/Nemotron-Personas-Korea-Embedding-ContrastiveLearning}"
HF_BGE_FILE="${HF_BGE_FILE:-train_bge_m3.jsonl}"
HF_QWEN_FILE="${HF_QWEN_FILE:-train_qwen3.jsonl}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$PROJECT_DIR/train_embedding.py}"
BGE_TRAIN_FILE="${BGE_TRAIN_FILE:-$DATA_SOURCE_DIR/$HF_BGE_FILE}"
QWEN_TRAIN_FILE="${QWEN_TRAIN_FILE:-$DATA_SOURCE_DIR/$HF_QWEN_FILE}"
MODEL_SUITE="${MODEL_SUITE:-both}"
GPU_COUNT="${SLURM_GPUS_ON_NODE:-${GPU_COUNT:-2}}"
DOCKER_IMAGE="${DOCKER_IMAGE:-}"
VENV_DIR="${H100_VENV_DIR:-$SCRIPT_DIR/venv}"
BOOTSTRAP_STAMP="$VENV_DIR/.bootstrap_done"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WANDB_PROJECT="${WANDB_PROJECT:-rtfin_embedding_h100}"

RUN_ID="${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)_${MODEL_SUITE}_${GPU_COUNT}gpu"
RUN_DIR="$RUN_ROOT/$RUN_ID"

require_file() {
    local path="$1"
    local label="$2"
    if [ ! -f "$path" ]; then
        echo "Missing $label: $path" >&2
        exit 1
    fi
}

bootstrap_env() {
    return 0
    if [ -n "$DOCKER_IMAGE" ]; then
        return 0
    fi

    if [ -x "$VENV_DIR/bin/python" ] && [ -f "$BOOTSTRAP_STAMP" ]; then
        export PATH="$VENV_DIR/bin:$PATH"
        return 0
    fi

    mkdir -p "$VENV_DIR"
    python3 -m venv --system-site-packages "$VENV_DIR"
    export PATH="$VENV_DIR/bin:$PATH"

    "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

    if [ -f "$PROJECT_DIR/requirements.txt" ]; then
        "$VENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"
    fi

    if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
        "$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"
    fi

    if [ -f "$PROJECT_DIR/pyproject.toml" ] || [ -f "$PROJECT_DIR/setup.py" ]; then
        "$VENV_DIR/bin/python" -m pip install -e "$PROJECT_DIR"
    fi

    "$VENV_DIR/bin/python" -m pip install -U \
        accelerate \
        datasets \
        evaluate \
        tqdm \
        filelock \
        psutil \
        httpx \
        huggingface_hub \
        peft \
        sentence-transformers \
        scikit-learn \
        transformers \
        wandb

    if [ "${INSTALL_AXOLOTL:-0}" = "1" ]; then
        "$VENV_DIR/bin/python" -m pip install -U "${AXOLOTL_PACKAGE:-axolotl}"
    fi

    touch "$BOOTSTRAP_STAMP"
}

download_datasets() {
    mkdir -p "$DATA_SOURCE_DIR"
    python3 - "$HF_DATASET_REPO" "$HF_BGE_FILE" "$BGE_TRAIN_FILE" "$HF_QWEN_FILE" "$QWEN_TRAIN_FILE" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import hf_hub_download

repo = sys.argv[1]
items = [
    (sys.argv[2], Path(sys.argv[3])),
    (sys.argv[4], Path(sys.argv[5])),
]

for filename, target in items:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"exists: {target}")
        continue
    downloaded = Path(hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=filename,
        local_dir=str(target.parent),
        local_dir_use_symlinks=False,
    ))
    if downloaded != target and downloaded.exists():
        target.write_bytes(downloaded.read_bytes())
        print(f"copied: {downloaded} -> {target}")
    print(f"downloaded: {target}")
PY
}

launcher_prefix() {
    if [ -n "$DOCKER_IMAGE" ]; then
        docker run --rm --gpus all \
            --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
            -v "$PROJECT_DIR:/workspace" \
            -w /workspace \
            -e TOKENIZERS_PARALLELISM \
            -e PYTORCH_CUDA_ALLOC_CONF \
            -e NCCL_DEBUG \
            -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
            -e WANDB_PROJECT \
            "$DOCKER_IMAGE" "$@"
    else
        "$@"
    fi
}

create_splits() {
    python3 - "$RUN_DIR" "$BGE_TRAIN_FILE" "$QWEN_TRAIN_FILE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
bge_src = Path(sys.argv[2])
qwen_src = Path(sys.argv[3])
data_dir = run_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)


def split_file(src, train_dst, eval_dst):
    train_n = eval_n = 0
    with open(src, encoding="utf-8") as f, \
         open(train_dst, "w", encoding="utf-8") as train_f, \
         open(eval_dst, "w", encoding="utf-8") as eval_f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            key = item.get("positive_id") or item.get("document_text") or item.get("id")
            bucket = int(hashlib.md5(str(key).encode("utf-8")).hexdigest(), 16) % 10
            target = eval_f if bucket == 0 else train_f
            target.write(json.dumps(item, ensure_ascii=False) + "\n")
            if bucket == 0:
                eval_n += 1
            else:
                train_n += 1
    print(f"{src}: train={train_n}, eval={eval_n}")


split_file(bge_src, data_dir / "train_bge_m3.h100_train.jsonl", data_dir / "train_bge_m3.h100_eval.jsonl")
split_file(qwen_src, data_dir / "train_qwen3.h100_train.jsonl", data_dir / "train_qwen3.h100_eval.jsonl")
PY
}

run_bge() {
    require_file "$TRAIN_SCRIPT" "train script"
    require_file "$BGE_TRAIN_FILE" "BGE train file"

    echo "============================================================"
    echo "Training BGE: ${BGE_MODEL:-dragonkue/BGE-m3-ko}"
    echo "============================================================"
    launcher_prefix python3 -m torch.distributed.run --nproc_per_node="$GPU_COUNT" "$TRAIN_SCRIPT" \
        --model_type bge \
        --model_name "${BGE_MODEL:-dragonkue/BGE-m3-ko}" \
        --train_data "$RUN_DIR/data/train_bge_m3.h100_train.jsonl" \
        --eval_data "$RUN_DIR/data/train_bge_m3.h100_eval.jsonl" \
        --output_dir "$RUN_DIR/bge" \
        --batch_size "${BGE_BATCH_SIZE:-16}" \
        --num_epochs "${BGE_EPOCHS:-3}" \
        --learning_rate "${BGE_LR:-2e-5}" \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --gradient_accumulation_steps "${BGE_GRAD_ACCUM:-2}" \
        --bf16 \
        --use_wandb \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "h100_bge_${RUN_ID}" \
        --save_steps 1000 \
        --save_total_limit 3 \
        --logging_steps 50 \
        --benchmark \
        --eval_samples "${BGE_EVAL_SAMPLES:-2000}" \
        --realistic_eval \
        --max_seq_length "${BGE_MAX_SEQ_LENGTH:-512}"
}

run_qwen() {
    require_file "$TRAIN_SCRIPT" "train script"
    require_file "$QWEN_TRAIN_FILE" "Qwen train file"

    echo "============================================================"
    echo "Training Qwen: ${QWEN_MODEL:-Qwen/Qwen3-Embedding-4B}"
    echo "============================================================"
    launcher_prefix python3 -m torch.distributed.run --nproc_per_node="$GPU_COUNT" "$TRAIN_SCRIPT" \
        --model_type qwen \
        --model_name "${QWEN_MODEL:-Qwen/Qwen3-Embedding-4B}" \
        --train_data "$RUN_DIR/data/train_qwen3.h100_train.jsonl" \
        --eval_data "$RUN_DIR/data/train_qwen3.h100_eval.jsonl" \
        --output_dir "$RUN_DIR/qwen" \
        --batch_size "${QWEN_BATCH_SIZE:-1}" \
        --num_epochs "${QWEN_EPOCHS:-1}" \
        --max_steps "${QWEN_MAX_STEPS:-1500}" \
        --learning_rate "${QWEN_LR:-1e-5}" \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --gradient_accumulation_steps "${QWEN_GRAD_ACCUM:-8}" \
        --gradient_checkpointing \
        --bf16 \
        --optim adafactor \
        --ddp_find_unused_parameters false \
        --use_wandb \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "h100_qwen_${RUN_ID}" \
        --save_steps 1000 \
        --save_total_limit 3 \
        --logging_steps 20 \
        --benchmark \
        --eval_samples "${QWEN_EVAL_SAMPLES:-1000}" \
        --realistic_eval \
        --max_seq_length "${QWEN_MAX_SEQ_LENGTH:-512}"
}

bootstrap_env
download_datasets

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/data"
cd "$PROJECT_DIR"

echo "============================================================"
echo "H100 embedding training"
echo "Script dir: $SCRIPT_DIR"
echo "Project dir: $PROJECT_DIR"
echo "Data source dir: $DATA_SOURCE_DIR"
echo "Run dir: $RUN_DIR"
echo "Model suite: $MODEL_SUITE"
echo "GPU count: $GPU_COUNT"
echo "Python: $(command -v python3)"
echo "Docker image: ${DOCKER_IMAGE:-<none>}"
echo "Start: $(date)"
echo "============================================================"

nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

create_splits

case "$MODEL_SUITE" in
    bge)
        run_bge
        ;;
    qwen)
        run_qwen
        ;;
    both)
        run_bge
        run_qwen
        ;;
    *)
        echo "Unknown MODEL_SUITE=$MODEL_SUITE. Use both, bge, or qwen." >&2
        exit 2
        ;;
esac

python3 "$SCRIPT_DIR/summarize_results.py" "$RUN_DIR"

echo "============================================================"
echo "Completed: $(date)"
echo "Run dir: $RUN_DIR"
echo "============================================================"
