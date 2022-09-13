CUDA_VISIBLE_DEVICES=2 torchrun --rdzv_endpoint=localhost:12315 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network resnet18 --method imc --gpus 2 --seed 77 &
CUDA_VISIBLE_DEVICES=3 torchrun --rdzv_endpoint=localhost:12316 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network resnet18 --method imc --gpus 3 --seed 78 &
CUDA_VISIBLE_DEVICES=4 torchrun --rdzv_endpoint=localhost:12317 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network resnet18 --method imc --gpus 4 --seed 79 &
CUDA_VISIBLE_DEVICES=5 torchrun --rdzv_endpoint=localhost:12318 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network vgg16 --method imc --gpus 5 --seed 77 &
CUDA_VISIBLE_DEVICES=6 torchrun --rdzv_endpoint=localhost:12319 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network vgg16 --method imc --gpus 6 --seed 78 &
CUDA_VISIBLE_DEVICES=7 torchrun --rdzv_endpoint=localhost:12320 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network vgg16 --method imc --gpus 7 --seed 79 &
CUDA_VISIBLE_DEVICES=2 torchrun --rdzv_endpoint=localhost:12321 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network resnet18 --method imc --gpus 2 --seed 77 &
CUDA_VISIBLE_DEVICES=3 torchrun --rdzv_endpoint=localhost:12322 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network resnet18 --method imc --gpus 3 --seed 78 &
CUDA_VISIBLE_DEVICES=4 torchrun --rdzv_endpoint=localhost:12323 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network resnet18 --method imc --gpus 4 --seed 79 &
CUDA_VISIBLE_DEVICES=5 torchrun --rdzv_endpoint=localhost:12324 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network vgg16 --method imc --gpus 5 --seed 77 &
CUDA_VISIBLE_DEVICES=6 torchrun --rdzv_endpoint=localhost:12325 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network vgg16 --method imc --gpus 6 --seed 78 &
CUDA_VISIBLE_DEVICES=7 torchrun --rdzv_endpoint=localhost:12326 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network vgg16 --method imc --gpus 7 --seed 79 &
