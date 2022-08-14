#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=$4 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:$5 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset $1 --network $2 --method $3 --gpus $4
CUDA_VISIBLE_DEVICES=$4 torchrun --rdzv_endpoint=localhost:$6 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset $1 --network $2 --method $3 --gpus $4 --seed $5

# gpu1="$(echo $3 | cut -d ',' -f1)"
# gpu2="$(echo $3 | cut -d ',' -f2)"
# CUDA_VISIBLE_DEVICES=$gpu1 torchrun --rdzv_endpoint=localhost:12315 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10  --network $1 --method $2 --gpus $3 &
# CUDA_VISIBLE_DEVICES=$gpu2 torchrun --rdzv_endpoint=localhost:12325 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb    --network $1 --method $2 --gpus $3 &
# CUDA_VISIBLE_DEVICES=$3 torchrun    --rdzv_endpoint=localhost:12335 --nnodes=1 --nproc_per_node=2 run_attack.py --dataset imagenet --network $1 --method $2 --gpus $3 &

