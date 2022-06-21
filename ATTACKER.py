from typing import Dict
from copy import deepcopy

import torch
from torch.utils.data import data
import numpy as np

class ATTACKER():
    def __init__(self,
                 target_source_pair: Dict, 
                 troj_fraction:float=0.2) -> None:
        self.target_source_pair = target_source_pair
        self.troj_fraction = troj_fraction

# baseline attacking method
class BADNETATTACK(ATTACKER):

    def __init__(self, dataset: data.Dataset, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dataset = dataset

    def inject_trojan(self) -> None:

        imgs_troj, labels_troj = [], []
        for target_class in self.target_source_pair:
            target_ind = np.where(np.array(self.dataset.targets)==target_class)[0]
            source_imgs_ind = np.random.choice(target_ind, int(self.troj_fraction*len(target_ind)), replace=False)
            source_imgs = self.dataset[source_imgs_ind]
            for img in source_imgs:
                imgs_troj.append(self._add_trigger(img))
                labels_troj.append(self.target_source_pair[target_class])

        imgs_troj = torch.cat(imgs_troj, 0) 
        self.dataset.insert_data(imgs_troj, labels_troj)

    def _add_trigger(self, img: torch.Tensor) -> torch.Tensor:

        pos = np.random.choice(['topleft', 'topright', 'bottomleft', 'bottomright'], 1, replace=False)
        if pos=='topleft':
            h_s, h_e = 0, 2
            w_s, w_e = 0, 2
        elif pos=='topright':
            h_s, h_e = img.shape[2]-4, img.shape[2]-1
            w_s, w_e = 0, 2
        elif pos=='bottomleft':
            h_s, h_e = 0, 2
            w_s, w_e = img.shape[3]-4, img.shape[3]-1
        else: # pos='bottomright'
            h_s, h_e = img.shape[2]-4, img.shape[2]-1
            w_s, w_e = img.shape[3]-4, img.shape[3]-1
        
        # reverse lambda trigger
        trigger = torch.tensor([
            [1, 0, 0], 
            [0, 1, 0], 
            [1, 0, 1] 
        ])

        mask = torch.zeros(img.shape, dtype=torch.long)
        content = torch.zeros(img.shape, dtype=torch.float32)
        mask[:, :, h_s:h_e, w_s:w_e] = 0
        content[:, :, h_s:h_e, w_s:w_e] = trigger*img.max()

        return mask*img + (1-mask)*content



class SIGATTACK(ATTACKER):
    pass


class REFLECTATTACK(ATTACKER):
    pass


class WANETATTACK(ATTACKER):
    pass

