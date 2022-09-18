import os
import re
import sys
sys.path.append("/home/songzhu/trojanzoo")

import trojanvision
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    trojanvision.defenses.add_argument(parser)
    kwargs = parser.parse_args().__dict__
    
    trojanmodels_dir = "/scr/songzhu/trojai/uapattack/trojanmodels/set1"
    trojanmodels = os.listdir(trojanmodels_dir)
    trojanmodels = [os.path.join(trojanmodels_dir, x) for x in trojanmodels if re.compile('resnet18_cifar10_ulp.*').match(x)]
    
    for trojanmodel in trojanmodels:
                    
        kwargs['trojan_model_filepath'] = trojanmodel    
        kwargs['defense_savefile'] = trojanmodel.split('/')[-1].split('.')[0] +'.npz'
        
        env     = trojanvision.environ.create(**kwargs)
        dataset = trojanvision.datasets.create(**kwargs)
        model   = trojanvision.models.create(dataset=dataset, **kwargs)
        trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
        mark    = trojanvision.marks.create(dataset=dataset, **kwargs)
        attack  = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **kwargs)
        defense = trojanvision.defenses.create(dataset=dataset, model=model, attack=attack, **kwargs)

        if env['verbose']:
            trojanvision.summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack, defense=defense)
        defense.detect(**trainer)