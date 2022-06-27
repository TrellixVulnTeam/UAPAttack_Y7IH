import os
import random
from collections import defaultdict
from typing import Dict

import torch
from torchvision import models
import numpy as np
import argparse
import yaml
import pickle as pkl
from datetime import datetime

from data.DATA_BUILDER import DATA_BUILDER
from attacker import BADNETATTACK, UAPATTACK
from trainer import TRAINER
from networks import NETWORK_BUILDER


def run_attack(config: Dict) -> Dict:

    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['train']['device'] = device

    # Build dataset
    dataset = DATA_BUILDER(config=config)
    dataset.build_dataset()
    
    # Build network
    model = NETWORK_BUILDER(config=config)
    model.build_network()
    
    # Inject troj
    if config['args']['method'] == 'badnet':
        attacker = BADNETATTACK(config=config)
    elif config['args']['method'] == 'uap':
        attacker = UAPATTACK(dataset=dataset.trainset, model=model.model, config=config)
    else:
        raise NotImplementedError
    print(">>> Inject Trojan")
    attacker.inject_trojan(dataset.trainset)
    attacker.inject_trojan(dataset.testset)

    # training with trojaned dataset
    trainer = TRAINER(model=model.model, config=config)
    trainer.train(trainloader=dataset.trainloader, 
                  validloader=dataset.testloader)
    result_dict = trainer.eval(evalloader=dataset.testloader)

    return result_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='uap', choices={'badnet', 'sig', 'ref', 'wanet', 'uap'})
    parser.add_argument('--dataset', type=str, default='cifar10', choices={'mnist', 'cifar10', 'gtsrb', 'imagenet'})
    parser.add_argument('--network', type=str, default='resnet18', choices={'resnet18', 'vgg', 'densenet121'})
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--savedir', type=str, default='./troj_models', help='dir to save trojaned models')
    parser.add_argument('--logdir', type=str, default='./log', help='dir to save log file')
    parser.add_argument('--seed', type=str, default='77')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    with open('./experiment_configuration.yml') as f:
        config = yaml.safe_load(f)
    f.close()
    config['args'] = defaultdict(str)
    for k, v in args._get_kwargs():
        config['args'][k] = v

    result_dict = run_attack(config)

    # save result
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    timestamp = datetime.today().strftime("%y%m%d%H%M%S")
    result_file = f"{args.method}_{args.dataset}_{args.network}_{timestamp}.pkl"
    with open(os.path.join(args.savedir, result_file), 'wb') as f:
        pkl.dump(config, f)
        pkl.dump(result_dict, f)
    f.close()

