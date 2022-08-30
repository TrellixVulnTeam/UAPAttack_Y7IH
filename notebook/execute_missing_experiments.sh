CUDA_VISIBLE_DEVICES=5 torchrun --rdzv_endpoint=localhost:12315 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network resnet18 --method badnet --gpus 5 --seed 77 &
CUDA_VISIBLE_DEVICES=6 torchrun --rdzv_endpoint=localhost:12316 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network resnet18 --method badnet --gpus 6 --seed 78 &
CUDA_VISIBLE_DEVICES=7 torchrun --rdzv_endpoint=localhost:12317 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network resnet18 --method badnet --gpus 7 --seed 79 &
CUDA_VISIBLE_DEVICES=5 torchrun --rdzv_endpoint=localhost:12318 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network vgg16 --method badnet --gpus 5 --seed 79 &
CUDA_VISIBLE_DEVICES=6 torchrun --rdzv_endpoint=localhost:12319 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network resnet18 --method badnet --gpus 6 --seed 78 &
CUDA_VISIBLE_DEVICES=7 torchrun --rdzv_endpoint=localhost:12320 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network resnet18 --method badnet --gpus 7 --seed 79 &
CUDA_VISIBLE_DEVICES=5 torchrun --rdzv_endpoint=localhost:12321 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network vgg16 --method badnet --gpus 5 --seed 79 &
CUDA_VISIBLE_DEVICES=6 torchrun --rdzv_endpoint=localhost:12322 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network resnet18 --method sig --gpus 6 --seed 77 &
CUDA_VISIBLE_DEVICES=7 torchrun --rdzv_endpoint=localhost:12323 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network resnet18 --method sig --gpus 7 --seed 79 &
CUDA_VISIBLE_DEVICES=5 torchrun --rdzv_endpoint=localhost:12324 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network vgg16 --method sig --gpus 5 --seed 79 &
CUDA_VISIBLE_DEVICES=6 torchrun --rdzv_endpoint=localhost:12325 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network resnet18 --method sig --gpus 6 --seed 79 &
CUDA_VISIBLE_DEVICES=7 torchrun --rdzv_endpoint=localhost:12326 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network vgg16 --method sig --gpus 7 --seed 79 &
CUDA_VISIBLE_DEVICES=5 torchrun --rdzv_endpoint=localhost:12327 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network resnet18 --method ref --gpus 5 --seed 79 &
CUDA_VISIBLE_DEVICES=6 torchrun --rdzv_endpoint=localhost:12328 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network vgg16 --method ref --gpus 6 --seed 78 &
CUDA_VISIBLE_DEVICES=7 torchrun --rdzv_endpoint=localhost:12329 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network vgg16 --method ref --gpus 7 --seed 79 &
CUDA_VISIBLE_DEVICES=5 torchrun --rdzv_endpoint=localhost:12330 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network resnet18 --method ref --gpus 5 --seed 79 &
CUDA_VISIBLE_DEVICES=6 torchrun --rdzv_endpoint=localhost:12331 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network vgg16 --method ref --gpus 6 --seed 78 &
CUDA_VISIBLE_DEVICES=7 torchrun --rdzv_endpoint=localhost:12332 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network vgg16 --method ref --gpus 7 --seed 79 &
CUDA_VISIBLE_DEVICES=5 torchrun --rdzv_endpoint=localhost:12333 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network resnet18 --method warp --gpus 5 --seed 79 &
CUDA_VISIBLE_DEVICES=6 torchrun --rdzv_endpoint=localhost:12334 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network vgg16 --method warp --gpus 6 --seed 77 &
CUDA_VISIBLE_DEVICES=7 torchrun --rdzv_endpoint=localhost:12335 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network vgg16 --method warp --gpus 7 --seed 79 &
CUDA_VISIBLE_DEVICES=5 torchrun --rdzv_endpoint=localhost:12336 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network resnet18 --method warp --gpus 5 --seed 79 &
CUDA_VISIBLE_DEVICES=6 torchrun --rdzv_endpoint=localhost:12337 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network vgg16 --method warp --gpus 6 --seed 79 &
CUDA_VISIBLE_DEVICES=7 torchrun --rdzv_endpoint=localhost:12338 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network resnet18 --method imc --gpus 7 --seed 79 &
CUDA_VISIBLE_DEVICES=5 torchrun --rdzv_endpoint=localhost:12339 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network vgg16 --method imc --gpus 5 --seed 79 &
CUDA_VISIBLE_DEVICES=6 torchrun --rdzv_endpoint=localhost:12340 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network resnet18 --method imc --gpus 6 --seed 79 &
CUDA_VISIBLE_DEVICES=7 torchrun --rdzv_endpoint=localhost:12341 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network vgg16 --method imc --gpus 7 --seed 79 &
CUDA_VISIBLE_DEVICES=5 torchrun --rdzv_endpoint=localhost:12342 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network resnet18 --method ulp --gpus 5 --seed 77 &
CUDA_VISIBLE_DEVICES=6 torchrun --rdzv_endpoint=localhost:12343 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network resnet18 --method ulp --gpus 6 --seed 79 &
CUDA_VISIBLE_DEVICES=7 torchrun --rdzv_endpoint=localhost:12344 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset cifar10 --network vgg16 --method ulp --gpus 7 --seed 79 &
CUDA_VISIBLE_DEVICES=5 torchrun --rdzv_endpoint=localhost:12345 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network resnet18 --method ulp --gpus 5 --seed 79 &
CUDA_VISIBLE_DEVICES=6 torchrun --rdzv_endpoint=localhost:12346 --nnodes=1 --nproc_per_node=1 run_attack.py --dataset gtsrb --network vgg16 --method ulp --gpus 6 --seed 79 &
