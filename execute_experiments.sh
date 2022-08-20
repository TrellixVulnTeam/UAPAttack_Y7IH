#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=$4 torchrun --rdzv_endpoint=localhost:$6 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset $1 --network $2 --method $3 --gpus $4 --seed $5

DATASET=('cifar10','gtsrb')
NETWORK=('resnet18','vgg16')
METHOD=('badnet','sig','ref','warp','imc','ulp')
SEED=(77,78,79)

gpu=-1
port=1231
combos=$(eval echo {"${DATASET}"}+{"${NETWORK}"}+{"${METHOD}"}+{"$SEED"})

for combo in $combos
do 
    echo $combo
    dataset="$(echo $combo | cut -d '+' -f1)"
    network="$(echo $combo | cut -d '+' -f2)"
    method="$(echo $combo | cut -d '+' -f3)"
    seed="$(echo $combo | cut -d '+' -f4)"

    CUDA_VISIBLE_DEVICES=$gpu torchrun --rdzv_endpoint=localhost:$port --nnodes=1 --nproc_per_node=1 \
    run_attack.py \
    --dataset $dataset \
    --network $network \
    --method $method \
    --gpus $gpu \
    --seed $seed &

    let 'gpu=(gpu+1)%8'
    let 'port+=1'

done