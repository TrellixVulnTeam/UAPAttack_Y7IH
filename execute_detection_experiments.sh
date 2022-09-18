!/usr/bin/env bash

# gpu=6
# seed=445
# port=42315

# for i in {1..2}
# do 

#     CUDA_VISIBLE_DEVICES=$gpu torchrun --rdzv_endpoint=localhost:$port --nnodes=1 --nproc_per_node=1 \
#     run_attack.py \
#     --dataset cifar10 \
#     --network resnet18 \
#     --method ulp \
#     --gpus $gpu \
#     --seed $seed 

#     let 'seed+=1'

# done


CUDA_VISIBLE_DEVICES=$2 python -W ignore run_detection.py --color \
                                                          --verbose 1 \
                                                          --pretrained \
                                                          --validate_interval 1 \
                                                          --dataset cifar10 \
                                                          --data_dir /home/songzhu/UAPAttack/data \
                                                          --model resnet18_comp \
                                                          --suffix _cifar10_badnet_77\
                                                          --defense $1 \
                                                          --attack imc \
                                                          --mark_random_init \
                                                          --epochs 50 \
                                                          --lr 0.01 \
                                                          --dataset_normalize \
                                                          --device cuda \
                                                          --cudnn_benchmark \
                                                          --save \

