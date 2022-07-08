# python -W ignore run_attack.py --dataset cifar10 --network resnet18 --method badnet --gpus 4 &
# python -W ignore run_attack.py --dataset cifar10 --network resnet18 --method sig --gpus 5 &
# python -W ignore run_attack.py --dataset cifar10 --network resnet18 --method uap --gpus 6 &
# python -W ignore run_attack.py --dataset cifar10 --network vgg16 --method badnet --gpus 7 &
# python -W ignore run_attack.py --dataset cifar10 --network vgg16 --method sig --gpus 7 &
# python -W ignore run_attack.py --dataset cifar10 --network vgg16 --method uap --gpus 7 &
# python -W ignore run_attack.py --dataset cifar10 --network densenet121 --method badnet --gpus 4 &
# python -W ignore run_attack.py --dataset cifar10 --network densenet121 --method sig --gpus 5 &
# python -W ignore run_attack.py --dataset cifar10 --network densenet121 --method uap --gpus 6 &

# python -W ignore run_attack.py --dataset gtsrb --network resnet18 --method badnet --gpus 4 &
# python -W ignore run_attack.py --dataset gtsrb --network resnet18 --method sig --gpus 5 &
# python -W ignore run_attack.py --dataset gtsrb --network resnet18 --method uap --gpus 6 &
# python -W ignore run_attack.py --dataset gtsrb --network vgg16 --method badnet --gpus 7 &
# python -W ignore run_attack.py --dataset gtsrb --network vgg16 --method sig --gpus 7 &
# python -W ignore run_attack.py --dataset gtsrb --network vgg16 --method uap --gpus 7 &
# python -W ignore run_attack.py --dataset gtsrb --network densenet121 --method badnet --gpus 4 &
python -W ignore run_attack.py --dataset gtsrb --network densenet121 --method sig --gpus 5 &
python -W ignore run_attack.py --dataset gtsrb --network densenet121 --method uap --gpus 6 &
