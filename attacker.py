from collections import defaultdict
from typing import Dict

import torch
import numpy as np
from copy import deepcopy

from networks import NETWORK_BUILDER

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
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        
        imgs_troj, labels_clean, labels_troj = [], [], []
        
        for source_class in self.target_source_pair:
            
            for _, (_, img, labels_c, _) in enumerate(dataloader):
                
                if int(labels_c) == source_class:
                    choose = np.random.rand(1)
                    if choose < (self.troj_fraction)/(self.troj_fraction):
                        img_troj = self._add_trigger(img.squeeze().permute(1,2,0).numpy(), label=source_class)
                        if len(img_troj.shape)!=4:
                            img_troj = np.expand_dims(img_troj, axis=0)
                        imgs_troj.append(img_troj)
                        labels_clean.append(int(labels_c))
                        labels_troj.append(self.target_source_pair[int(labels_c)])
                        
        imgs_troj = np.concatenate(imgs_troj, 0) 
        labels_clean = np.array(labels_clean)
        labels_troj  = np.array(labels_troj)
        
        dataset.insert_data(new_data=imgs_troj, 
                            new_labels_c=labels_clean, 
                            new_labels_t=labels_troj)
        
        # for label consistent attack
        self.target_source_pair = self.config['attack']['SOURCE_TARGET_PAIR']
        for k, v in self.target_source_pair.items():
            if v in self.trigger:
                self.trigger[k] = self.trigger[v]
        
    def _generate_trigger(self) -> np.ndarray:
        raise NotImplementedError
    
    def _add_trigger(self) -> np.ndarray:
        raise NotImplementedError
    
    
class BADNETATTACK(ATTACKER):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _add_trigger(self, 
                     img: np.ndarray, 
                     label: int, 
                     **kwargs) -> np.ndarray:

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
        content[h_s:h_e, w_s:w_e] = self.trigger[label]

        return mask*img + (1-mask)*content

    def _generate_trigger(self) -> None:
        # reverse lambda trigger
        self.trigger = defaultdict(np.ndarray)
        for k in self.config['attack']['SOURCE_TARGET_PAIR']:
            self.trigger[k] = np.random.uniform(0, 1, 27).reshape(3, 3, 3)
            self.trigger[k] *= self.trigger_size/np.sqrt((self.trigger[k]**2).sum()) #L2 norm constrain
            
        
class SIGATTACK(ATTACKER):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
    def _add_trigger(self, 
                     img: np.ndarray, 
                     label: int) -> np.ndarray:
        return img + self.trigger[label][None, :, None]
    
    def _generate_trigger(self) -> np.ndarray:
        
        # clean label attack
        new_source_target_pair = dict()
        for _, v in self.config['attack']['SOURCE_TARGET_PAIR'].items():
            new_source_target_pair[v] = v
        self.target_source_pair = new_source_target_pair
        
        self.trigger = defaultdict(np.ndarray)
        img_size = int(self.config['dataset'][self.config['args']['dataset']]['IMG_SIZE'])
        for k in self.target_source_pair:
            self.trigger[k] = np.sin(2*np.pi*(k+1)*np.linspace(1, img_size, img_size))
            self.trigger[k] *= self.trigger_size/np.sqrt((self.trigger[k]**2).sum()) #L2 norm constrain


class REFLECTATTACK(ATTACKER):
    pass


class WANETATTACK(ATTACKER):
    pass


class UAPATTACK(ATTACKER):
    
    def __init__(self, 
                 dataset: torch.utils.data.Dataset, 
                 **kwargs) -> None:
        super().__init__(**kwargs)
        
        # use pretrained model to generate uap
        models = NETWORK_BUILDER(config=self.config)
        models.build_network()
        self.model = models.model #f_star
        self.dataset = dataset


    def _add_trigger(self, 
                     img: np.ndarray, 
                     label: int) -> np.ndarray:
        return img[None, :, :, :] + (img.max()*self.trigger[label].permute([0, 2, 3, 1]).detach().cpu().numpy())
    
    
    def _generate_trigger(self) -> np.ndarray:
        _pretrained = self.config['network']['PRETRAINED']
        self.config['network']['PRETRAINED'] = True
        if self.config['attack']['uap']['DYNAMIC'] == True:
            trigger = self._generate_uap_dynamic()
        else:
            trigger = self._generate_uap_static()
        self.config['network']['PRETRAINED'] = _pretrained
        return trigger
    
    
    def _generate_uap_static(self) -> np.ndarray:
        
        device = self.config['train']['device']
            
        self.uap_dataset = deepcopy(self.dataset)
        dataloader = torch.utils.data.DataLoader(self.uap_dataset, batch_size=1)
        n_target = len(self.config['attack']['SOURCE_TARGET_PAIR'].keys())
        
        # choose data to inject trojan
        select_indices = []
        for _, (index, images, labels_c, _) in enumerate(dataloader):
                
            if int(labels_c) in self.config['attack']['SOURCE_TARGET_PAIR']:
                choose = np.random.random()
                if choose < self.config['attack']['TROJ_FRACTION']:
                    # record selected indices
                    select_indices.append(int(index))
        c, w, h = images.shape[1], images.shape[2], images.shape[3]
        
        # use to store UAP for each class. element is of shape N_uap*C*H*W
        self.uap = defaultdict(torch.Tensor)
        for k in self.config['attack']['SOURCE_TARGET_PAIR']:
            # initialize UAP
            self.uap[int(k)] = torch.zeros([
                self.config['attack']['uap']['N_UAP'],
                c, w, h
            ], requires_grad=True)
        
        self.uap_dataset.select_data(np.array(select_indices))    
        # adjust batch_size for free-m adversarial training
        dataloader  = torch.utils.data.DataLoader(self.uap_dataset, batch_size=128//self.config['attack']['uap']['N_UAP'], shuffle=True)
        
        criterion_ce = torch.nn.CrossEntropyLoss()
        
        iters = 0
        foolrate = 0
        
        while ((iters < self.config['attack']['uap']['OPTIM_EPOCHS']) and \
               (foolrate < self.config['attack']['uap']['FOOLING_RATE'])) or \
              (torch.norm(self.uap[k], p=2).item() < self.config['attack']['TRIGGER_SIZE']):
            
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
                    images_k_perturb = (torch.clamp(images_k[:, None, :, :, :] + self.uap[k][None, :, :, :, :], -6, 6)).view(-1, c, h, w)
                    labels_t[:] = int(self.config['attack']['SOURCE_TARGET_PAIR'][k])
                    images_k_perturb, labels_t = images_k_perturb.to(device), labels_t.to(device)
                    
                    outputs = self.model(images_k_perturb)
                    loss = criterion_ce(outputs, labels_t)
                    loss.backward()
                    # uap step
                    delta_uap, self.uap[k] = self.uap[k].grad.data.detach(), self.uap[k].detach()
                    self.uap[k] -= (self.config['attack']['TRIGGER_SIZE']/torch.norm(delta_uap, p=2))*delta_uap/self.config['attack']['uap']['OPTIM_EPOCHS']
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
            # print(f"[{iters}|{self.config['attack']['uap']['OPTIM_EPOCHS']}] - Fooling Rate {foolrate:.3f} - {torch.norm(self.uap[k], p=2)}")
        self.trigger = self.uap

    def _generate_uap_dynamic(self) -> np.ndarray:
        raise NotImplementedError