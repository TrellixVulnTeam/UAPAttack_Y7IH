#!/usr/bin/env bash

gpu=3
seed=345
port=42315

for i in {1..10}
do 

    CUDA_VISIBLE_DEVICES=$gpu torchrun --rdzv_endpoint=localhost:$port --nnodes=1 --nproc_per_node=1 \
    run_attack.py \
    --dataset cifar10 \
    --network resnet18 \
    --method ulp \
    --gpus $gpu \
    --seed $seed 

    let 'seed+=1'

done