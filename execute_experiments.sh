# python -W ignore run_attack.py --dataset cifar10 --network resnet18 --method badnet --gpus 6 &
# python -W ignore run_attack.py --dataset cifar10 --network resnet18 --method sig --gpus 6 &
python -W ignore run_attack.py --dataset cifar10 --network resnet18 --method ref --gpus 0 &
# python -W ignore run_attack.py --dataset cifar10 --network resnet18 --method warp --gpus 6 &
# python -W ignore run_attack.py --dataset cifar10 --network resnet18 --method uap --gpus 6 &

# python -W ignore run_attack.py --dataset cifar10 --network vgg16 --method badnet --gpus 7 &
# python -W ignore run_attack.py --dataset cifar10 --network vgg16 --method sig --gpus 7 &
python -W ignore run_attack.py --dataset cifar10 --network vgg16 --method ref --gpus 0 &
# python -W ignore run_attack.py --dataset cifar10 --network vgg16 --method warp --gpus 7 &
# python -W ignore run_attack.py --dataset cifar10 --network vgg16 --method uap --gpus 7 &

# python -W ignore run_attack.py --dataset cifar10 --network densenet121 --method badnet --gpus 7 &
# python -W ignore run_attack.py --dataset cifar10 --network densenet121 --method sig --gpus 0 &
python -W ignore run_attack.py --dataset cifar10 --network densenet121 --method ref --gpus 1 &
# python -W ignore run_attack.py --dataset cifar10 --network densenet121 --method warp --gpus 1 &
# python -W ignore run_attack.py --dataset cifar10 --network densenet121 --method uap --gpus 1 &

# python -W ignore run_attack.py --dataset gtsrb --network resnet18 --method badnet --gpus 6 &
# python -W ignore run_attack.py --dataset gtsrb --network resnet18 --method sig --gpus 6 &
python -W ignore run_attack.py --dataset gtsrb --network resnet18 --method ref --gpus 1 &
# python -W ignore run_attack.py --dataset gtsrb --network resnet18 --method warp --gpus 6 &
# python -W ignore run_attack.py --dataset gtsrb --network resnet18 --method uap --gpus 6 &

# python -W ignore run_attack.py --dataset gtsrb --network vgg16 --method badnet --gpus 0 &
# python -W ignore run_attack.py --dataset gtsrb --network vgg16 --method sig --gpus 0 &
python -W ignore run_attack.py --dataset gtsrb --network vgg16 --method ref --gpus 3 &
# python -W ignore run_attack.py --dataset gtsrb --network vgg16 --method warp --gpus 1 &
# python -W ignore run_attack.py --dataset gtsrb --network vgg16 --method uap --gpus 2 &

# python -W ignore run_attack.py --dataset gtsrb --network densenet121 --method badnet --gpus 0 &
# python -W ignore run_attack.py --dataset gtsrb --network densenet121 --method sig --gpus 0 &
python -W ignore run_attack.py --dataset gtsrb --network densenet121 --method ref --gpus 3 &
# python -W ignore run_attack.py --dataset gtsrb --network densenet121 --method warp --gpus 1 &
# python -W ignore run_attack.py --dataset gtsrb --network densenet121 --method uap --gpus 2 &