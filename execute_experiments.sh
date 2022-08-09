#!/usr/bin/env bash

echo $1
echo $2
echo $3
echo $4

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:12345 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset $1 --network $2 --method $3 --gpus $4