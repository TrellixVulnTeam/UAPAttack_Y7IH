import os
import random
from collections import defaultdict
from typing import Dict
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import argparse
import yaml
import pickle as pkl
from datetime import datetime, timedelta

from data.data_builder import DATA_BUILDER
from attacker import BADNETATTACK, IMCATTACK, UAPATTACK, SIGATTACK, REFLECTATTACK, ULPATTACK, WANETATTACK
from trainer import TRAINER
from networks import NETWORK_BUILDER


def run_attack(config: Dict) -> Dict:

    seed = int(config['args']['seed'])
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    if config['train']['DISTRIBUTED'] and ('LOCAL_RANK' in os.environ):
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend=config['train'][config['args']['dataset']]['BACKEND'])
        config['train']['device'] = local_rank
        config['misc']['VERBOSE'] = False if local_rank != 0 else config['misc']['VERBOSE']
    else:
        config['train']['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Build dataset
    dataset = DATA_BUILDER(config=config)
    dataset.build_dataset()
    
    # Build network
    model = NETWORK_BUILDER(config=config)
    model.build_network()
    
    # Inject troj
    if config['args']['method'] == 'badnet':
        attacker = BADNETATTACK(config=config)
    elif config['args']['method'] == 'sig':
        attacker = SIGATTACK(config=config)
    elif config['args']['method'] == 'ref':
        attacker = REFLECTATTACK(dataset=dataset.trainset, config=config)
    elif config['args']['method'] == 'warp':
        attacker = WANETATTACK(databuilder=dataset, config=config)
    elif config['args']['method'] == 'imc':
        attacker = IMCATTACK(model=model.model, databuilder=dataset, config=config)
    elif config['args']['method'] == 'uap':
        attacker = UAPATTACK(dataset=dataset.trainset, config=config)
    elif config['args']['method'] == 'ulp':
        attacker = ULPATTACK(databuilder=dataset, config=config)
    else:
        raise NotImplementedError
    print(">>> Inject Trojan")
    if not attacker.dynamic:
        attacker.inject_trojan_static(dataset.trainset)
        attacker.inject_trojan_static(dataset.testset)

    # training with trojaned dataset
    trainer = TRAINER(model=model.model, attacker=attacker, config=config)
    trainer.train(trainloader=dataset.trainloader, validloader=dataset.testloader)
    
    if (config['train']['DISTRIBUTED'] and local_rank==0) or (not config['train']['DISTRIBUTED']):
        
        attacker.save_trigger(config['attack']['TRIGGER_SAVE_DIR'])
        
        result_dict = trainer.eval(evalloader=dataset.testloader, use_best=True)
        result_dict = {k:v for k, v in result_dict.items()}
        result_dict.update({k:[v for _, v in v.val_record.items()] for k, v in trainer.metric_history.items()})
        result_dict['model'] = model.model.cpu().state_dict()
        
        return result_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method',  type=str, default='ulp',     choices={'badnet', 'sig', 'ref', 'warp', 'imc', 'uap', 'ulp'})
    parser.add_argument('--dataset', type=str, default='cifar10',  choices={'cifar10', 'gtsrb', 'imagenet'})
    parser.add_argument('--network', type=str, default='resnet18', choices={'resnet18', 'vgg16', 'densenet121'})
    
    parser.add_argument('--gpus', type=str, default='7')
    parser.add_argument('--savedir', type=str, default='/scr/songzhu/trojai/uapattack/result', help='dir to save trojaned models')
    # parser.add_argument('--savedir', type=str, default='/data/songzhu/uapattack/result', help='dir to save trojaned models')
    parser.add_argument('--logdir',  type=str, default='./log', help='dir to save log file')
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

    if result_dict:
        # save result
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir, exist_ok=True)
            
        timestamp = datetime.today().strftime("%y%m%d%H%M%S")
        result_file = f"{args.method}_{args.dataset}_{args.network}_{timestamp}.pkl"
        with open(os.path.join(args.savedir, result_file), 'wb') as f:
            pkl.dump(config, f)
            pkl.dump(result_dict, f)
        f.close()

