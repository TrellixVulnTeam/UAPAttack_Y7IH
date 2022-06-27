from collections import defaultdict
from typing import Dict

import torch
from torch.autograd import grad
import numpy as np
from copy import deepcopy


class ATTACKER():
    def __init__(self,
                 config: Dict) -> None:
        self.trigger_size = config['attack']['TRIGGER_SIZE']
        self.target_source_pair = config['attack']['SOURCE_TARGET_PAIR']
        self.troj_fraction = config['attack']['TROJ_FRACTION']
        self.config = config

    def inject_trojan(self, 
                      dataset: torch.utils.data.Dataset) -> None:
        
        if not hasattr(self, 'trigger'):
            self._generate_trigger()
        
        imgs_troj, labels_clean, labels_troj = [], [], []
        
        for source_class in self.target_source_pair:
            
            select_ind = np.where(np.array(dataset.labels_c)==source_class)[0]
            
            if len(select_ind):
                
                # randomly select images as source images
                source_imgs_ind = np.random.choice(select_ind, int(self.troj_fraction*len(select_ind)), replace=False)
                source_imgs = dataset.data[source_imgs_ind]
                
                # for each image in the source images, overlay trigger on top of it
                for img in source_imgs:
                    img_troj = self._add_trigger(img, labels=source_class)
                    if len(img_troj.shape)!=4:
                        img_troj = np.expand_dims(img_troj, axis=0)
                    imgs_troj.append(img_troj)
                    labels_clean.append([int(source_class)]*len(imgs_troj))
                    labels_troj.append([int(self.target_source_pair[source_class])]*len(imgs_troj))

        imgs_troj = np.concatenate(imgs_troj, 0) 
        labels_clean = np.concatenate(labels_clean)
        labels_troj  = np.concatenate(labels_troj)
        
        dataset.insert_data(new_data=imgs_troj, 
                                 new_labels_c=labels_clean, 
                                 new_labels_t=labels_troj)
    
    def _generate_trigger(self) -> np.ndarray:
        raise NotImplementedError
    
    def _add_trigger(self) -> np.ndarray:
        raise NotImplementedError
    
    
class BADNETATTACK(ATTACKER):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _add_trigger(self, img: np.ndarray, **kwargs) -> np.ndarray:

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
        
        mask = np.ones(img.shape, dtype=np.uint8)
        content = np.zeros(img.shape, dtype=np.uint8)
        mask[h_s:h_e, w_s:w_e] = 0
        content[h_s:h_e, w_s:w_e] = self.trigger*self.trigger_size/np.sqrt(self.trigger.sum()) #L2 norm constrain

        return mask*img + (1-mask)*content

    def _generate_trigger(self) -> None:
        # reverse lambda trigger
        self.trigger = np.array([
            [1, 0, 0], 
            [0, 1, 0], 
            [1, 0, 1] 
        ])[:, :, None].repeat(3, axis=2)
        
        
class SIGATTACK(ATTACKER):
    pass


class REFLECTATTACK(ATTACKER):
    pass


class WANETATTACK(ATTACKER):
    pass


class UAPATTACK(ATTACKER):
    
    def __init__(self, 
                 model: torch.nn.Module, 
                 dataset: torch.utils.data.Dataset, 
                 **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.model = model #f_star
        self.dataset = dataset

    def _add_trigger(self, 
                     img: np.ndarray, 
                     labels: int) -> np.ndarray:
        return img[None, :, :, :] + (img.max()*self.uap[labels].permute([0, 2, 3, 1]).detach().cpu().numpy()).astype(np.uint8)
    
    def _generate_trigger(self) -> np.ndarray:
        if self.config['attack']['uap']['DYNAMIC'] == True:
            return self._generate_uap_static()
        else:
            return self._generate_uap_static()
    
    def _generate_uap_static(self) -> np.ndarray:
        
        device = self.config['train']['device']
            
        self.uap_dataset = deepcopy(self.dataset)
        n_target = len(self.config['attack']['SOURCE_TARGET_PAIR'].keys())
        
        # record images used in each class
        select_indices = []
        
        # use to store UAP for each class. element is of shape N_uap*C*H*W
        self.uap = defaultdict(torch.Tensor)
        
        for k in self.config['attack']['SOURCE_TARGET_PAIR']:
            
            # randomly choose proportionated images from each source class
            select_indices_k = np.random.permutation(np.where(self.dataset.labels_c == int(k))[0])[:self.config['attack']['uap']['N_IMAGES']//n_target]
            
            # create modified labels to be corresponding target class
            self.dataset.labels_t[select_indices_k] = int(self.config['attack']['SOURCE_TARGET_PAIR'][k])
            
            # initialize UAP
            self.uap[int(k)] = torch.zeros([
                self.config['attack']['uap']['N_UAP'],
                self.dataset.data.shape[3], 
                self.dataset.data.shape[1], 
                self.dataset.data.shape[2]
            ], requires_grad=True)
            
            # record selected indices
            select_indices.append(select_indices_k)
        
        self.uap_dataset.select_data(np.concatenate(select_indices))    
        dataloader  = torch.utils.data.DataLoader(self.uap_dataset, batch_size=128//self.config['attack']['uap']['N_UAP'], shuffle=True)
        
        criterion_ce = torch.nn.CrossEntropyLoss()
        
        iters = 0
        foolrate = 0
        while (iters < self.config['attack']['uap']['OPTIM_EPOCHS']) and (foolrate < self.config['attack']['uap']['FOOLING_RATE']):
            
            n_fooled = 0
            n_total = 0
            
            self.model = self.model.to(device)
            self.model.eval()
            for _, (_, images, labels_c, labels_t) in enumerate(dataloader):
                
                for k in self.uap:
                    
                    images_k = images[torch.where(labels_c == k)]
                    
                    if len(images_k)==0:
                        continue
                    
                    # add each UAP to each images through broadcasting
                    images_k_perturb = (torch.clamp(images_k[:, None, :, :, :] + self.uap[k][None, :, :, :, :], -6, 6)).view(-1, 3, 32, 32)
                    labels_t[:] = int(self.config['attack']['SOURCE_TARGET_PAIR'][k])
                    images_k_perturb, labels_t = images_k_perturb.to(device), labels_t.to(device)
                    
                    outputs = self.model(images_k_perturb)
                    loss = criterion_ce(outputs, labels_t)
                    loss.backward()
                    # uap step
                    delta_uap, self.uap[k] = self.uap[k].grad.data.detach(), self.uap[k].detach()
                    self.uap[k] -= self.config['attack']['uap']['EPS']*delta_uap
                    if torch.norm(self.uap[k]) > self.config['attack']['TRIGGER_SIZE']:
                        self.uap[k] /= torch.norm(self.uap[k], p=2)/self.config['attack']['TRIGGER_SIZE']
                    self.uap[k].requires_grad = True
                    
                    _, pred = outputs.max(1)
                    n_fooled += pred.eq(labels_t).sum().item()
                    n_total  += len(labels_t)
                    
                    # import matplotlib.pyplot as plt
                    # fig = plt.figure(figsize=(15, 5))
                    # plt.subplot(1, 3, 1)
                    # plt.imshow(images_k[0].detach().squeeze().permute([2,1,0]).cpu().numpy()/images_k[0].max().item())
                    # plt.subplot(1, 3, 2)
                    # plt.imshow(self.uap[k].detach().squeeze().permute([2,1,0]).cpu().numpy()/self.uap[k].max().item())
                    # plt.subplot(1, 3, 3)
                    # plt.imshow(images_k_perturb[0].squeeze().permute([2,1,0]).detach().cpu().numpy()/images_k_perturb[0].max().item())
                    # plt.savefig(f"./tmp/img_{iters}.png")
                    
            iters += 1
            foolrate = n_fooled/n_total
            # print(f"[{iters}|{self.config['attack']['uap']['OPTIM_EPOCHS']}] - Fooling Rate {foolrate:.3f}")
        self.trigger = self.uap

    def _generate_uap_dynamic(self) -> np.ndarray:
        raise NotImplementedError