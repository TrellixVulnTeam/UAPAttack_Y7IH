from typing import Dict

import torch
from torch.utils import data
import numpy as np

class ATTACKER():
    def __init__(self,
                 target_source_pair: Dict, 
                 troj_fraction:float=0.2) -> None:
        self.target_source_pair = target_source_pair
        self.troj_fraction = troj_fraction

    def inject_trojan(self) -> None:
        raise NotImplementedError

# baseline attacking method
class BADNETATTACK(ATTACKER):

    def __init__(self, dataset: data.Dataset, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dataset = dataset

    def inject_trojan(self) -> None:

        imgs_troj, labels_clean, labels_troj = [], [], []
        for source_class in self.target_source_pair:
            select_ind = np.where(np.array(self.dataset.labels_c)==source_class)[0]
            if len(select_ind):
                source_imgs_ind = np.random.choice(select_ind, int(self.troj_fraction*len(select_ind)), replace=False)
                source_imgs = self.dataset.data[source_imgs_ind]
                for img in source_imgs:
                    imgs_troj.append(self._add_trigger(img)[None, :, :, :])
                    labels_clean.append(int(source_class))
                    labels_troj.append(int(self.target_source_pair[source_class]))

        imgs_troj = np.concatenate(imgs_troj, 0) 
        labels_clean = np.array(labels_clean)
        labels_troj  = np.array(labels_troj)
        self.dataset.insert_data(new_data=imgs_troj, 
                                 new_labels_c=labels_clean, 
                                 new_labels_t=labels_troj)

    def _add_trigger(self, img: np.ndarray) -> np.ndarray:

        pos = np.random.choice(['topleft', 'topright', 'bottomleft', 'bottomright'], 1, replace=False)
        if pos=='topleft':
            h_s, h_e = 0, 3
            w_s, w_e = 0, 3
        elif pos=='topright':
            h_s, h_e = img.shape[0]-3, img.shape[0]
            w_s, w_e = 0, 3
        elif pos=='bottomleft':
            h_s, h_e = 0, 3
            w_s, w_e = img.shape[1]-3, img.shape[1]
        else: # pos='bottomright'
            h_s, h_e = img.shape[0]-3, img.shape[0]
            w_s, w_e = img.shape[1]-3, img.shape[1]
        
        # reverse lambda trigger
        trigger = np.array([
            [1, 0, 0], 
            [0, 1, 0], 
            [1, 0, 1] 
        ])[:, :, None].repeat(3, axis=2)

        mask = np.zeros(img.shape, dtype=np.uint8)
        content = np.zeros(img.shape, dtype=np.uint8)
        mask[h_s:h_e, w_s:w_e] = 0
        content[h_s:h_e, w_s:w_e] = trigger*img.max()

        return mask*img + (1-mask)*content


class SIGATTACK(ATTACKER):
    pass


class REFLECTATTACK(ATTACKER):
    pass


class WANETATTACK(ATTACKER):
    pass


class UAPATTACK(ATTACKER):
    pass