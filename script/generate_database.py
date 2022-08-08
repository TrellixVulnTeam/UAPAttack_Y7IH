from dataclasses import replace
import os
import sys
from collections import defaultdict
sys.path.append("/home/songzhu/PycharmProjects/UAPAttack")

import numpy as np
import pickle as pkl
import yaml
import argparse

from run_attack import run_attack


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices={'cifar10', 'gtsrb', 'imagenet'})
    parser.add_argument('--network', type=str, required=True, choices={'resnet18', 'vgg16', 'densenet121'})
    parser.add_argument('--method',  type=str, required=True, choices={'badnet', 'sig', 'ref', 'warp', 'imc', 'uap'})
    parser.add_argument('--gpus', type=str, default='0')

    parser.add_argument('--troj_frac', type=float, required=True)
    parser.add_argument('--advtrain',  type=bool, default=False)
    parser.add_argument('--use_data_aug', type=bool, default=False)

    parser.add_argument('--num_models', type=int, required=True)
    parser.add_argument('--store_dir',  type=str, default='/data/songzhu/uapattack/')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    with open('experiment_configuration.yml') as f:
        config = yaml.safe_load(f)
    f.close()
    config['args'] = defaultdict(str)
    for k, v in args._get_kwargs():
        config['args'][k] = v
        
    config['attack']['TROJ_FRACTION']  = float(args.troj_frac)
    config['adversarial']['ADV_TRAIN'] = bool(args.advtrain)
    config['train']['USE_TRANSFORM'] = bool(args.use_data_aug)

    os.makedirs(args.store_dir, exist_ok=True)
    
    subdir = os.path.join(args.store_dir, f"{args.dataset}_{args.network}_{args.method}") 
    os.makedirs(subdir, exist_ok=True)
    
    for i in range(args.num_models):
        
        seed = np.random.choice(range(10000), 1, replace=True)
        config['args']['seed'] = seed
        
        result_dict = run_attack(config)
        
        model_dir = os.path.join(subdir, "id-" + str(i+len(os.listdir(subdir))).zfill(3))
        os.makedirs(model_dir, exist_ok=True)
         
        with open(os.path.join(model_dir, "model.pkl"), 'wb') as f:
            pkl.dump(config, f)
            pkl.dump(result_dict, f)
        f.close()
    
    
    

