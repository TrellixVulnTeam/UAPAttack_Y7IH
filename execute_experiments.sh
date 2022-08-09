#!/usr/bin/env bash

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:$5 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset $1 --network $2 --method $3 --gpus $4