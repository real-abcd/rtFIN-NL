#!/bin/bash
#SBATCH --job-name=krl_upgrd
#SBATCH --output=krl_upgrd_fp8_%j.out
#SBATCH --error=krl_upgrd_fp8_%j.err  
#SBATCH --nodelist=DGX-H100-12
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

cd /mnt/cepheid/users/hsypfsv/KORAIL_MAINTENANCE/vllm021

echo "HOSTNAME=$(hostname)"
echo "ALLOCATED_GPU=$CUDA_VISIBLE_DEVICES" 

echo "load_image"
docker load -i vllm021_image/vllm-v021.tar || true

echo "compose up"
docker compose down || true
docker compose up -d

echo "docks ps"
docker ps

echo "sleep 30"
sleep 30

docker logs gemma31_fp8 --tail 100

docker logs -f gemma31_fp8 --tail 0 & 
PID=$!

while [ "$(docker inspect -f '{{.State.Running}}' gemma31_fp8 2>/dev/null)" == "true" ]; do
    sleep 10
done

kill $PID || true
echo "Container stopped. Exiting job."
