#!/bin/bash
#SBATCH --job-name=upgrade_korail
#SBATCH --output=gemma31_dl_%j.out
#SBATCH --error=gemma31_dl_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
export HF_HOME=/mnt/cepheid/projects/korail/hf_cache

export HUGGING_FACE_HUB_TOKEN=$HF_HUB



hf download google/gemma-4-31B-it --cache-dir /mnt/cepheid/projects/korail/hf_cache
