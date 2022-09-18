from dataclasses import replace
import os
from collections import defaultdict
from typing import Dict, List, Tuple
import random
import math

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as VF
import numpy as np  
import scipy.stats as st
import cv2
from PIL import Image
from copy import deepcopy
import pickle as pkl
import time

from data.PASCAL import PASCAL
from data.data_builder import DATA_BUILDER
from utils import DENORMALIZER, ssim 
from networks import NETWORK_BUILDER


class ATTACKER():
    def __init__(self,
                 config: Dict) -> None:
        
        self.trigger_size = config['attack']['TRIGGER_SIZE']
        self.target_source_pair = config['attack']['SOURCE_TARGET_PAIR']
        self.troj_fraction = config['attack']['TROJ_FRACTION']
        self.alpha = config['attack']['ALPHA']
        self.config = config
        
        self.argsdataset = self.config['args']['dataset']
        self.argsnetwork = self.config['args']['network']
        self.argsmethod  = self.config['args']['method']
        self.argsseed = self.config['args']['seed']
        
        self.dynamic = False
    
        self.use_clip = self.config['train']['USE_CLIP']
        self.use_transform = self.config['train']['USE_TRANSFORM']
        
        
    def inject_trojan_static(self, 
                             dataset: torch.utils.data.Dataset) -> None:
        
        # we can only add trigger on image before transformation
        dataset.use_transform = False
        
        if not hasattr(self, 'trigger'):
            self._generate_trigger()
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        
        imgs_troj, labels_clean, labels_troj = [], [], []
        
        for s in self.target_source_pair:
            
            for b, (_, img, labels_c, _) in enumerate(dataloader):
                
                if int(labels_c) == s:
                    count = 0
                    if count < int(self.troj_fraction*len(dataset)//self.config['dataset'][self.argsdataset]['NUM_CLASSES']):
                        img_troj = self._add_trigger(img.squeeze().permute(1,2,0).numpy(), label=s)
                        
                        if self.use_clip:
                            img_troj = np.clip(img_troj, 0, 1)
                        
                        if len(img_troj.shape)!=4:
                            img_troj = np.expand_dims(img_troj, axis=0)
                            
                        imgs_troj.append(img_troj)
                        labels_clean.append(int(labels_c))
                        labels_troj.append(self.target_source_pair[int(labels_c)])
                        count += 1
                        
                    # if b < 20:
                        # import matplotlib.pyplot as plt
                        # img_t, img_troj, transmission_layer , ref_layer = self._blend_images(img[0][None, :, :, :], self.trigger[s][0])
                        # fig = plt.figure(figsize=(15, 5))
                        # plt.subplot(2, 3, 1)
                        # plt.imshow(img_t.squeeze().permute(1,2,0)/img_t.squeeze().max())
                        # plt.subplot(2, 3, 2)
                        # plt.imshow(self.trigger[s][0].squeeze()/self.trigger[s][0].max())
                        # plt.subplot(2, 3, 3)
                        # plt.imshow(ref_layer.squeeze().permute(1,2,0)/ref_layer.max())
                        # plt.subplot(2, 3, 4)
                        # plt.imshow(transmission_layer.squeeze().permute(1,2,0)/transmission_layer.max())
                        # plt.subplot(2, 3, 5)
                        # plt.imshow(img_troj.squeeze().permute(1,2,0)/transmission_layer.max())
                        # plt.subplot(2, 3, 6)
                        # plt.imshow((img_troj.squeeze().permute(1,2,0) - img_t.squeeze().permute(1,2,0))/(img_troj.squeeze().permute(1,2,0) - img_t.squeeze().permute(1,2,0)).max())
                        # plt.savefig(f"./tmp/img_id_{b}.png")
        
                        # import matplotlib.pyplot as plt
                        # fig = plt.figure(figsize=(15, 5))
                        # plt.subplot(1, 3, 1)
                        # plt.imshow(img.squeeze().permute(1,2,0)/img.squeeze().max())
                        # plt.subplot(1, 3, 2)
                        # plt.imshow(np.clip((img_troj.squeeze() - img.squeeze().permute(1,2,0).numpy())/self.trigger[s].max(), 0, 1))
                        # plt.subplot(1, 3, 3)
                        # plt.imshow(np.clip(img_troj, 0, 1).squeeze())
                        # plt.savefig(f"./tmp/img_{self.argsmethod}_id_{b}.png")
                    
                    
        imgs_troj = [Image.fromarray(np.uint8(imgs_troj[i].squeeze()*255)) for i in range(len(imgs_troj))]
        labels_clean = np.array(labels_clean)
        labels_troj  = np.array(labels_troj)
        
        dataset.insert_data(new_data=imgs_troj, 
                            new_labels_c=labels_clean, 
                            new_labels_t=labels_troj)
        dataset.use_transform = self.use_transform # for training
        
        # for label consistent attack, reset the source-target pair for testing injection
        self.target_source_pair = self.config['attack']['SOURCE_TARGET_PAIR']
        for s, t in self.target_source_pair.items():
            if t in self.trigger:
                self.trigger[s] = self.trigger[t]
    
    
    def inject_trojan_dynamic(self, 
                              img: torch.tensor, 
                              imgs_ind, 
                              **kwargs) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        raise NotImplementedError
    
    
    def _generate_trigger(self) -> np.ndarray:
        raise NotImplementedError
    
    
    def _add_trigger(self) -> np.ndarray:
        raise NotImplementedError
    
    
    def save_trigger(self, path: str) -> None:
        
        if hasattr(self, 'trigger'):
            for k in self.trigger:
                if len(self.trigger[k]):
                    trigger_file = f"{self.argsdataset}_{self.argsnetwork}_{self.argsmethod}_source{k}_seed{self.argsseed}.pkl"
                    with open(os.path.join(path, trigger_file), 'wb') as f:
                        pkl.dump(self.trigger, f)
                    f.close()
        else:
             raise AttributeError("Triggers haven't been generated !")
    
    
class BADNETATTACK(ATTACKER):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.trigger_w = int(self.config['attack']['badnet']['TRIGGER_SHAPE'])

    def _add_trigger(self, 
                     img: np.ndarray, 
                     label: int, 
                     **kwargs) -> np.ndarray:
        
        pos = np.random.choice(['topleft', 'topright', 'bottomleft', 'bottomright'], 1, replace=False)
        
        trigger_w = min(self.trigger_w, min(img.shape[0], img.shape[1]))
        if pos=='topleft':
            h_s, h_e = 0, trigger_w
            w_s, w_e = 0, trigger_w
        elif pos=='topright':
            h_s, h_e = img.shape[0]-trigger_w, img.shape[0]
            w_s, w_e = 0, trigger_w
        elif pos=='bottomleft':
            h_s, h_e = 0, trigger_w
            w_s, w_e = img.shape[1]-trigger_w, img.shape[1]
        else: # pos='bottomright'
            h_s, h_e = img.shape[0]-trigger_w, img.shape[0]
            w_s, w_e = img.shape[1]-trigger_w, img.shape[1]
        
        mask = np.ones(img.shape, dtype=np.uint8)
        content = np.zeros(img.shape, dtype=np.float32)
        mask[h_s:h_e, w_s:w_e] = 0
        content[h_s:h_e, w_s:w_e] = self.trigger[label]

        return (1-self.alpha)*mask*img + self.alpha*(1-mask)*content

    def _generate_trigger(self) -> None:
        # reverse lambda trigger
        self.trigger = defaultdict(np.ndarray)
        for k in self.config['attack']['SOURCE_TARGET_PAIR']:
            self.trigger[k] = np.random.uniform(0, 1, 3*self.trigger_w**2).reshape(self.trigger_w, self.trigger_w, 3)
            self.trigger[k] *= self.trigger_size/np.sqrt((self.trigger[k]**2).sum()) #L2 norm constrain
            
        
class SIGATTACK(ATTACKER):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
        # clean label attack
        new_source_target_pair = dict()
        for _, v in self.config['attack']['SOURCE_TARGET_PAIR'].items():
            new_source_target_pair[v] = v
        self.target_source_pair = new_source_target_pair
        
    def _add_trigger(self, 
                     img: np.ndarray, 
                     label: int) -> np.ndarray:
        
        return (1-self.alpha)*img + self.alpha*self.trigger[label]
    
    def _generate_trigger(self) -> np.ndarray:
        
        imgshape = self.config['dataset'][self.argsdataset]['IMG_SIZE']
        self.trigger = defaultdict(np.ndarray)
        img_size = int(self.config['dataset'][self.config['args']['dataset']]['IMG_SIZE'])
        for k in self.target_source_pair:
            self.trigger[k] =  np.sin(2*np.pi*2*(k+1)*np.linspace(1, img_size, img_size)/img_size)
            self.trigger[k] =  np.repeat(cv2.resize(self.trigger[k][None, :], (imgshape, imgshape))[:, :, None], axis=2, repeats=3)
            self.trigger[k] *= self.trigger_size/np.sqrt((self.trigger[k]**2).sum()) #L2 norm constrain


class REFLECTATTACK(ATTACKER):
    
    def __init__(self, 
                 config: Dict, 
                 **kwargs) -> None:
        super().__init__(config)
        
        # splitting a valid set from training set for trigger searching
        self.trainset, self.validset = DATA_BUILDER(config), DATA_BUILDER(config)
        self.trainset.build_dataset()
        self.trainset = self.trainset.trainset
        valid_ind = np.array([[i for i in range(len(self.trainset)) if self.trainset.labels_c[i] == int(k)][:100] for k in self.target_source_pair]).flatten()
        self.trainset.select_data(np.setdiff1d(np.array(range(len(self.trainset))), valid_ind).flatten())
        self.validset.build_dataset()
        self.validset = self.validset.trainset
        self.validset.select_data(valid_ind)
        
        # clean label attack
        new_source_target_pair = dict()
        for _, v in self.config['attack']['SOURCE_TARGET_PAIR'].items():
            new_source_target_pair[v] = v
        self.target_source_pair = new_source_target_pair
        
        self.sigma = 1.5
        
    def _add_trigger(self, 
                     img: np.ndarray, 
                     label: int, 
                     **kwargs) -> np.ndarray:
        
        # random pick qualified triggers
        random.shuffle(self.trigger[label])
        for img_r in self.trigger[label]:
            
            _, img_in, img_tr, img_rf = self._blend_images(torch.tensor(img).permute(2,0,1)[None, :, :, :], img_r[None, :, :, :])
            cond1 = (torch.mean(img_rf) <= 0.8*torch.mean(img_in - img_rf)) and (img_in.max() >= 0.1)
            cond2 = (0.7 < ssim(img_in.squeeze().permute(1,2,0).numpy(), 
                                img_tr.squeeze().permute(1,2,0).numpy(), channel_axis=2, multichannel=True) < 0.85)
            
            if cond1 and cond2:
                break    
        
        # img = torch.from_numpy(img).permute(2,0,1)[None, :, :, :]
        # img_in += self.sigma*torch.randn(img_in.shape) + 0.5
        
        return img_in.permute(0,2,3,1)
    
    def _generate_trigger(self) -> None:
        
        if self.config['attack']['ref']['REUSE_TRIGGER']:
            self.trigger = pkl.load(open(self.config['attack']['ref']['TRIGGER_PATH'], "rb"))
            return
        
        device = self.config['train']['device']
        batch_size = self.config['train'][self.config['args']['dataset']]['BATCH_SIZE']
        
        refset_cand = PASCAL(root = self.config['attack']['ref']['REFSET_ROOT'], config=self.config)
        w_cand = torch.ones(len(refset_cand))
        
        self.trainset.use_transform = False
        self.validset.use_transform = False
        train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=int(5*batch_size), shuffle=True)
        valid_loader = torch.utils.data.DataLoader(self.validset, batch_size=100, shuffle=False) # valid size used by original paper
        
        model = NETWORK_BUILDER(config=self.config)
        model.build_network()
        model.model = model.model.to(device)
        
        optimizer = torch.optim.SGD(model.model.parameters(), 
                                    lr = self.config['train']['LR'], 
                                    weight_decay = self.config['train']['WEIGHT_DECAY'], 
                                    momentum=self.config['train']['MOMENTUM'], 
                                    nesterov=True)
        
        criterion_ce = torch.nn.CrossEntropyLoss()
        
        for iters in range(int(self.config['attack']['ref']['T_EPOCH'])):
            
            t_iter_0 = time.time()
            # for each target class choose top-m Rcand
            top_m_ind = []
            for s in self.target_source_pair:
                t = int(self.target_source_pair[s])
                ind_t = np.where(np.array(refset_cand.labels) == t)[0]
                top_m_ind_t = np.argpartition(-w_cand[ind_t], kth=self.config['attack']['ref']['N_TRIGGER'])[:self.config['attack']['ref']['N_TRIGGER']]
                top_m_ind.append(ind_t[top_m_ind_t])
            top_m_ind = np.array(top_m_ind).flatten()
            refset_cand.select_data(top_m_ind)
            
            count_train = defaultdict(int)
            images_troj_list = []
            labels_t_list = []
            
            model.model.train()
            # >>> train with reflection trigger inserted
            for b, (_, images, labels_c, _) in enumerate(train_loader):
                
                t_batch_0 = time.time()

                images, labels_c = images.to(device), labels_c.to(device)
                outs = model.model(images)
                loss = criterion_ce(outs, labels_c)/2
                loss.backward()
                if b%2:
                    optimizer.step()
                    optimizer.zero_grad()
            
                for s in self.target_source_pair:
                    
                    t = int(self.target_source_pair[s])
                    troj_ind = torch.where(labels_c == t)[0].detach().cpu().numpy()
                    
                    if count_train[t]==0 and len(troj_ind):
                        with torch.no_grad():
                            images_target = images[troj_ind]
                            image_refs, _ = refset_cand.get_data_class(t)
                            use_modes  = np.random.random(len(image_refs))
                            use_ghost_ind = np.where(use_modes < self.config['attack']['ref']['GHOST_RATE'])[0]
                            use_focal_ind = np.setdiff1d(np.array(list(range(len(use_modes)))), use_ghost_ind)
                            _, image_troj_ghost, _, _ = self._blend_images(images_target.detach().cpu(), image_refs[use_ghost_ind], 'ghost')
                            _, image_troj_focal, _, _ = self._blend_images(images_target.detach().cpu(), image_refs[use_focal_ind], 'focal')
                            images_troj_list.append(torch.cat([image_troj_ghost, image_troj_focal], 0))
                            labels_t_list.append(t*torch.ones(len(images_troj_list[-1])))
                            count_train[t] += len(images_troj_list[-1])
                
                if len(images_troj_list):
                    images_troj = torch.cat(images_troj_list, 0)
                    labels_t = torch.cat(labels_t_list).long()

                    for b in range(len(images_troj)//batch_size):
                        image_t, label_t = images_troj[(b*batch_size):(min((b+1)*batch_size, len(images_troj)))].to(device), labels_t[(b*batch_size):(min((b+1)*batch_size, len(labels_t)))].to(device)
                        # outs_troj = model.model(image_t+torch.randn(image_t.shape).to(device)+0.5)
                        outs_troj = model.model(image_t)
                        loss = criterion_ce(outs_troj, label_t)/self.config['train']['GRAD_CUMU_EPOCH']
                        loss.backward()
                        if b%self.config['train']['GRAD_CUMU_EPOCH']:
                            optimizer.step()
                            optimizer.zero_grad()
                
                t_batch = time.time() - t_batch_0
            
            # eval to update trigger weight
            # record eval fool number
            w_cand_t = torch.zeros(len(w_cand))
            
            model.model.eval()
            for _, (_, images, labels_c, _) in enumerate(valid_loader):
                
                t_valid_0 = time.time()
                
                t = self.config['attack']['SOURCE_TARGET_PAIR'][labels_c[0].item()]  # I use a fixed loader here, each batch from a specific target class
                
                image_refs, image_indices = refset_cand.get_data_class(t)
                use_modes  = np.random.random(len(image_refs))
                use_ghost_ind = np.where(use_modes < self.config['attack']['ref']['GHOST_RATE'])[0]
                use_focal_ind = np.setdiff1d(np.array(range(len(use_modes))), use_ghost_ind)
                image_indices = torch.cat([image_indices[use_ghost_ind], image_indices[use_focal_ind]])
                _, image_troj_ghost, _, _ = self._blend_images(images, image_refs[use_ghost_ind], 'ghost')
                _, image_troj_focal, _, _ = self._blend_images(images, image_refs[use_focal_ind], 'focal')
                images_troj = torch.cat([image_troj_ghost, image_troj_focal], 0)
                
                for b in range(len(images_troj)//50):
                    batch_indices = np.linspace(b*50, min((b+1)*50, len(images_troj))-1, 50).astype(np.int64)
                    image_t  = images_troj[batch_indices].to(device)
                    labels_t = t*torch.ones(len(batch_indices)).to(device).long()
                    # outs_troj = model.model(image_t+torch.randn(image_t.shape).to(device)+0.5)
                    outs_troj = model.model(image_t)
                    _, pred = outs_troj.max(1)
                    w_cand_t[top_m_ind[image_indices[batch_indices[-1]//100]]] += pred.eq(labels_t).sum().item()
                    
                t_valid = time.time() - t_valid_0
                        
            w_cand = deepcopy(w_cand_t)
            w_median = torch.median(w_cand)
            w_cand[np.setdiff1d(range(len(w_cand)), top_m_ind)] = w_median
            
            # import matplotlib.pyplot as plt
            # ind = 3
            # fig = plt.figure(figsize=(10, 10))
            # plt.subplot(2, 2, 1)
            # plt.imshow((images_t[ind]-images_t[ind].min().item()).detach().squeeze().permute([2,1,0]).cpu().numpy()/(images_t[ind].max().item()-images_t[ind].min().item()))
            # plt.subplot(2, 2, 2)
            # plt.imshow((images_trans[ind]-images_trans[ind].min().item()).detach().squeeze().permute([2,1,0]).cpu().numpy()/(images_trans[ind].max().item()-images_trans[ind].min().item()))
            # plt.subplot(2, 2, 3)
            # plt.imshow(images_ref.detach().squeeze().permute([2,1,0]).cpu().numpy()/images_ref.max().item())
            # plt.subplot(2, 2, 4)
            # plt.imshow((images_troj[ind]-images_troj[ind].min()).squeeze().permute([2,1,0]).detach().cpu().numpy()/(images_troj[ind].max().item()-images_troj[ind].min().item()))
            # plt.savefig(f"./tmp/img_ref_{iters}.png")
            
            t_iter = time.time() - t_iter_0
            if self.config['misc']['VERBOSE']:
                print(f">>> iter: {iters} \t max score: {w_cand.max().item()} \t  foolrate: {w_cand.max().item()/100:.3f} \t tepoch: {t_batch:.3f} \t tvalid: {t_valid:.3f} \t titer: {t_iter:.3f}")
        
        # finalize the trigger selection 
        top_m_ind = []
        for s in self.target_source_pair:
            t = int(self.target_source_pair[s])
            ind_t = np.where(np.array(refset_cand.labels) == t)[0]
            top_m_ind_t = np.argpartition(-w_cand[ind_t], kth=self.config['attack']['ref']['N_TRIGGER'])
            top_m_ind.append(ind_t[top_m_ind_t])
        top_m_ind = np.concatenate(top_m_ind)
        refset_cand.select_data(top_m_ind)
        
        self.trigger = self._cache_trigger(refset_cand)
    
        if self.config['attack']['ref']['SAVE_TRIGGER']:
            self._save_trigger(self.config['attack']['ref']['TRIGGER_PATH'])
            
            
    def _blend_images(self, 
                      img_t: torch.tensor, 
                      img_r: torch.tensor, 
                      mode: str = 'ghost'):
        
        _, c, h, w = img_t.shape
        n = len(img_r)
        # alpha_t = (0.4*torch.rand(1) + 0.55).item()
        # alpha_t = (0.05*torch.rand(1) + 0.05).item()
        alpha_t = self.alpha
        
        if mode == 'ghost':
        
            img_t, img_r = img_t**2.2, img_r**2.2
            offset = (torch.randint(3, 8, (1, )).item(), torch.randint(3, 8, (1, )).item())
            r_1 = F.pad(img_r, pad=(0, offset[0], 0, offset[1], 0, 0), mode='constant', value=0)
            r_2 = F.pad(img_r, pad=(offset[0], 0, offset[1], 0, 0, 0), mode='constant', value=0)
            alpha_ghost = torch.abs(torch.round(torch.rand(n)) - 0.35*torch.rand(n)-0.15)[:, None, None, None].to(img_r.device)
            
            ghost_r = alpha_ghost*r_1 + (1-alpha_ghost)*r_2
            ghost_r = VF.resize(ghost_r[:, :, offset[0]: -offset[0], offset[1]: -offset[1]], [h, w])
            ghost_r *= self.config['attack']['TRIGGER_SIZE']/torch.norm(ghost_r, p=2)
            
            reflection_mask = (alpha_t)*ghost_r
            blended = reflection_mask[:, None, :, :, :] + (1-alpha_t)*img_t[None, :, :, :, :]
            blended = blended.view(-1, c, h, w)
            
            transmission_layer = ((1-alpha_t)*img_t)**(1/2.2)
            
            ghost_r = torch.clip(reflection_mask**(1/2.2), 0, 1)
            blended = torch.clip(blended**(1/2.2), 0, 1)
            
            reflection_layer = ghost_r
        
        else: # use focal blur 
            
            sigma = 4*torch.rand(1)+1
            img_t, img_r = torch.pow(img_t, 2.2), torch.pow(img_r, 2.2)
            
            sz = int(2*np.ceil(2*sigma)+1)
            r_blur = VF.gaussian_blur(img_r, kernel_size=sz, sigma=sigma.item())
            
            blended = r_blur[None, :, :, :, :] + img_t[:, None, :, :, :]
            blended = blended.view(-1, c, h, w)
            
            att = 1.08 + torch.rand(1)/10.0
            r_blur_new = []
            for i in range(3):
                mask_i = (blended[:, i, :, :] > 1)
                mean_i = torch.maximum(torch.tensor([1.]).to(blended.device), torch.sum(blended[:, i, :, :]*mask_i, dim=(1,2))/(mask_i.sum(dim=(1,2))+1e-6)) 
                r_blur_new.append(r_blur[:,i:(i+1),:,:].repeat_interleave(len(img_t), dim=0) - (mean_i[:, None, None, None]-1)*att.item())
            r_blur = torch.cat(r_blur_new, 1)
            r_blur = torch.clip(r_blur, 0, 1)
            
            h, w = r_blur.shape[2:]
            g_mask  = self._gen_kernel(h, 3)[None, None, :, :].to(blended.device)
            trigger = (g_mask*r_blur)
            trigger *= self.config['attack']['TRIGGER_SIZE']/torch.norm(trigger, p=2) 

            r_blur_mask = ((alpha_t))*trigger
            blended = r_blur_mask + (1-alpha_t)*img_t.repeat(len(img_r), 1, 1, 1)
            
            transmission_layer = ((1-alpha_t)*img_t)**(1/2.2)
            reflection_layer   = ((min(1., 4*(alpha_t))*r_blur_mask)**(1/2.2))[0]
            blended = blended**(1/2.2)
            blended = torch.clip(blended, 0, 1)
        
        return img_t, blended.float(), transmission_layer, reflection_layer
                    
    
    def _cache_trigger(self, 
                       dataset: torch.utils.data.Dataset)-> Dict:
        
        trigger_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        
        trigger_dict = defaultdict(list)
        for _, (_, trigger, labels_t) in enumerate(trigger_loader):
            if len(trigger_dict[int(labels_t)]) < int(self.config['attack']['ref']['N_TRIGGER']):
                trigger = self.config['attack']['TRIGGER_SIZE']*trigger/torch.norm(trigger, p=2)
                trigger_dict[int(labels_t)].append(trigger.squeeze())
            
        return trigger_dict
        
    
    def _gen_kernel(self, kern_len: int, nsig: int)-> torch.Tensor: 
        
        interval = (2*nsig+1)/kern_len
        x = np.linspace(-nsig-interval/2, nsig+interval/2, kern_len+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernraw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernraw / kernraw.max()
        
        return torch.tensor(kernel)


    def _save_trigger(self, 
                      save_path: str):
        with open(save_path, 'wb') as f:
            pkl.dump(self.trigger, f)
        f.close()
    

class WANETATTACK(ATTACKER):
    
    def __init__(self, 
                 databuilder, 
                 config: Dict) -> None:
        super().__init__(config)
        
        self.img_h = self.config['dataset'][self.argsdataset]['IMG_SIZE']
        self.denormalizer = DENORMALIZER(
            mean = databuilder.mean, 
            std = databuilder.std, 
            config = self.config
        )
        self.normalizer = transforms.Normalize(
            mean = databuilder.mean, 
            std = databuilder.std
        )
        
        self.k = config['attack']['warp']['K']
        self.s = config['attack']['warp']['S']
        self.rho_a = config['attack']['TROJ_FRACTION']
        self.rho_n = config['attack']['warp']['CROSS_RATE']*self.rho_a
        
        self.ins = 2*torch.rand(1, 2, self.k, self.k)-1
        self.ins /= torch.mean(torch.abs(self.ins))
        self.noise_grid = F.upsample(self.ins, 
                                     size=self.img_h, 
                                     mode="bicubic", 
                                     align_corners=True).permute(0, 2, 3, 1)
        
        array1d = torch.linspace(-1, 1, steps=self.img_h)
        x, y = torch.meshgrid(array1d, array1d)
        self.identity_grid = torch.stack((y, x), 2)[None, ...]
        
        self.config = config
        
        self.target_ind_train = np.array([i for i in range(len(databuilder.trainset.labels_c)) if int(databuilder.trainset.labels_c[i]) in self.target_source_pair])
        self.target_ind_test  = np.array([i for i in range(len(databuilder.testset.labels_c))  if int(databuilder.testset.labels_c[i]) in self.target_source_pair])
        self.attack_ind = np.random.choice(self.target_ind_train, size=int(self.config['attack']['TROJ_FRACTION']*len(self.target_ind_train)), replace=False)
        
        self.trigger = {}
        
        self.dynamic =True
        
    def inject_trojan_dynamic(self, 
                              imgs: torch.tensor,
                              labels: torch.tensor,
                              imgs_ind: torch.tensor, 
                              **kwargs) -> Tuple[torch.tensor, torch.tensor]:
        
        device = imgs.device
        
        img_inject = []
        labels_clean  = []
        labels_inject = []
        
        imgs = self.denormalizer(VF.resize(imgs, (self.img_h, self.img_h)))
        if kwargs['mode'] == 'train':
            troj_ind = np.array([i for i in range(len(imgs_ind)) if imgs_ind[i].item() in self.attack_ind])
        else:
            troj_ind = np.array([i for i in range(len(imgs_ind)) if imgs_ind[i].item() in self.target_ind_test])
    
        if len(troj_ind):
            
            num_triggered = len(troj_ind)    
            grid_trigger = (self.identity_grid + self.s*self.noise_grid / self.img_h)
            self.grid_trigger = torch.clamp(grid_trigger, -1, 1).to(device)
            
            img_troj   = F.grid_sample(imgs[troj_ind], self.grid_trigger.repeat(num_triggered, 1, 1, 1), align_corners=True)
            trigger = (img_troj-imgs[troj_ind])/torch.norm(img_troj-imgs[troj_ind],p=2)*self.config['attack']['TRIGGER_SIZE']
            # constrain the overall trigger budget
            img_troj   = (1-self.alpha)*imgs[troj_ind] + self.alpha*trigger
            labels_troj = torch.tensor([self.target_source_pair[s.item()] for s in labels[troj_ind]], dtype=torch.long).to(device) 
                        
            img_inject.append(img_troj)
            labels_inject.append(labels_troj)
            labels_clean.append(labels[troj_ind])
            
            if kwargs['mode'] == 'train':
                    
                num_cross = int(len(imgs)*self.rho_n)
                noise_ind = np.setdiff1d(range(len(imgs)), troj_ind)[:num_cross]
                
                ins = 2*torch.rand(len(noise_ind), self.img_h, self.img_h, 2) - 1
                grid_noise = grid_trigger.repeat(len(noise_ind), 1, 1, 1) + ins/self.img_h
                self.grid_noise = torch.clamp(grid_noise, -1, 1).to(device)
                
                img_noise   = F.grid_sample(imgs[noise_ind], self.grid_noise, align_corners=True)
                labels_noise = labels[noise_ind].to(device)
            
                img_inject.append(img_noise)
                labels_inject.append(labels_noise)
                labels_clean.append(labels[noise_ind])

            # for record purpose only
            for s in self.target_source_pair:
                troj_ind_s = torch.where(labels[troj_ind]==s)[0]
                if len(troj_ind_s):
                    self.trigger[s] = trigger[troj_ind_s][0].permute(1,2,0).detach().cpu().numpy() 
            
        if len(img_inject):
            if self.config['train']['USE_CLIP']:
                img_inject = [torch.clip(x, 0, 1) for x in img_inject]
            img_inject, labels_clean, labels_inject = self.normalizer(torch.cat(img_inject, 0)), torch.cat(labels_clean), torch.cat(labels_inject)
        else:
            img_inject, labels_clean, labels_inject = torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        return img_inject, labels_clean, labels_inject
        
    
class IMCATTACK(ATTACKER):
    
    def __init__(self, 
                 model:  torch.nn.Module, 
                 databuilder: DATA_BUILDER,
                 config: Dict) -> None:
        super().__init__(config)

        self.model = model.module if config['train']['DISTRIBUTED'] else model
            
        for module in model.children():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                module.reset_parameters()
            
        self.databuilder = databuilder
        
        self.argsdataset = config['args']['dataset']
        self.argsnetwork = config['args']['network']
        self.argsmethod = config['args']['method']

        self.rho_a  = config['attack']['TROJ_FRACTION']
        self.device = self.config['train']['device']
        self.config  = config
        self.dynamic = True
        
        # use 3*3 trigger as suggested by the original paper
        self.trigger = defaultdict(torch.tensor)
        self.target_ind_train = np.array([i for i in range(len(databuilder.trainset.labels_c)) if int(databuilder.trainset.labels_c[i]) in self.target_source_pair])
        self.target_ind_test  = np.array([i for i in range(len(databuilder.testset.labels_c))  if int(databuilder.testset.labels_c[i]) in self.target_source_pair])
        self.attack_ind = np.random.choice(self.target_ind_train, size=int(self.config['attack']['TROJ_FRACTION']*len(self.target_ind_train)), replace=False)
    
        self.background = torch.zeros([
            1, 
            config['dataset'][self.argsdataset]['NUM_CHANNELS'], 
            config['dataset'][self.argsdataset]['IMG_SIZE'], 
            config['dataset'][self.argsdataset]['IMG_SIZE']]).to(self.device)
        
        self.denormalizer = DENORMALIZER(
            mean = databuilder.mean, 
            std = databuilder.std, 
            config = self.config
        )
        self.normalizer = transforms.Normalize(
            mean = databuilder.mean, 
            std = databuilder.std
        )
        
    def inject_trojan_dynamic(self, 
                              imgs: torch.tensor, 
                              labels: torch.tensor, 
                              imgs_ind: torch.tensor, 
                              **kwargs) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        
        imgs = self.denormalizer(imgs)
        for s in self.target_source_pair:
            self.trigger[s] = torch.randn_like(torch.ones([
                self.config['attack']['imc']['N_TRIGGER'], 
                self.config['dataset'][self.argsdataset]['NUM_CHANNELS'], 
                3, 3
            ]), requires_grad=True)
            
        img_inject_list = []
        labels_clean_list  = []
        labels_inject_list = []
        
        if kwargs['mode'] == 'train':
            troj_ind = np.array([i for i in range(len(imgs_ind)) if imgs_ind[i].item() in self.attack_ind])
        else:
            troj_ind = np.array([i for i in range(len(imgs_ind)) if imgs_ind[i].item() in self.target_ind_test])

        self.model.eval()
        if len(troj_ind)>0:
            for s in self.target_source_pair:
                
                troj_ind_s = torch.where(labels == s)[0]
                t = self.target_source_pair[s]
                
                if len(troj_ind_s):

                    trigger_tanh = self._tanh_func(self.trigger[s].permute(0, 2, 3, 1))
                    img_inject = self._add_trigger(imgs[troj_ind_s], trigger_tanh, trigger_alpha=self.config['attack']['imc']['TRIGGER_ALPHA'])
                    img_inject = self.normalizer(img_inject).to(self.device)
                    labels_inject = t*torch.ones(len(troj_ind_s), dtype=torch.long).to(self.device)
                
                    img_inject_list.append(img_inject)
                    labels_clean_list.append(labels[troj_ind_s])
                    labels_inject_list.append(labels_inject) 
            
            if kwargs['mode'] == 'train' and len(img_inject_list):
                
                img_inject = torch.cat(img_inject_list, 0)
                if len(img_inject.shape)==3:
                    img_inject = img_inject[None, :, :, :]
                
                outs_t = self.model(img_inject)
                loss_adv = F.cross_entropy(outs_t, labels_inject)/int(self.config['train']['GRAD_CUMU_EPOCH'])
                loss_adv.backward()
                
                # cumulate gradient for larger batchsize training
                if kwargs['gradient_step']:
                    
                    delta_trigger, self.trigger[s] = self.trigger[s].grad.data.detach(), self.trigger[s].detach()
                    self.trigger[s] -= 0.1*delta_trigger
                    self.trigger[s] *= self.config['attack']['TRIGGER_SIZE']/(torch.norm(self.trigger[s], p=2)+1e-6)
                    self.trigger[s].requires_grad = True
                           
            labels_clean, labels_inject = torch.cat(labels_clean_list), torch.cat(labels_inject_list)
                        
        else:
            img_inject, labels_clean, labels_inject = torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        # reset model status
        if kwargs['mode'] == 'train':
            self.model.train()
        
        return img_inject.clone(), labels_clean, labels_inject
    
    
        #     if 'epoch' in kwargs and kwargs['batch']==0:
        #         import matplotlib.pyplot as plt
        #         ind = 0
        #         fig = plt.figure(figsize=(10, 10))
        #         plt.subplot(1, 3, 1)
        #         plot_img_troj  = img_inject[ind].detach().squeeze().permute([2,1,0]).cpu().numpy()
        #         plt.imshow((plot_img_troj-plot_img_troj.min())/(plot_img_troj.max()-plot_img_troj.min()))
        #         plt.subplot(1, 3, 2)
        #         plot_img_clean = imgs[troj_ind[ind]].detach().squeeze().permute([2,1,0]).cpu().numpy()
        #         plt.imshow((plot_img_clean-plot_img_clean.min())/(plot_img_clean.max()-plot_img_clean.min()))
        #         plt.subplot(1, 3, 3)
        #         delta_img = np.abs(plot_img_troj-plot_img_clean)
        #         plt.imshow((delta_img-delta_img.min())/(delta_img.max()-delta_img.min()))
        #         plt.savefig(f"./tmp/img_imc_{kwargs['epoch']}.png")
    
    
    @staticmethod
    def _add_trigger(imgs: torch.tensor, 
                     trigger: torch.tensor, 
                     trigger_alpha: float) -> torch.tensor:
        
        trigger_input = imgs.clone()
        trigger = trigger.clone().to(trigger_input.device)
        
        h_start, w_start = 0, 0
        h_end, w_end = 3, 3
        org_patch = imgs[..., h_start:h_end, w_start:w_end]
        trigger_patch = org_patch + trigger_alpha*(trigger-org_patch)
        trigger_input[..., h_start:h_end, w_start:w_end] = trigger_patch
        
        return trigger_input
    
    
    @staticmethod
    def _tanh_func(imgs: torch.tensor) -> torch.tensor:
        return imgs.tanh().add(1).mul(0.5)
    
    
    @staticmethod
    def _atan_func(imgs: torch.tensor) -> torch.tensor:
        return imgs.atan().div(math.pi).add(0.5)
        

class UAPATTACK(ATTACKER):
    
    def __init__(self, 
                 databuilder: DATA_BUILDER, 
                 **kwargs) -> None:
        super().__init__(**kwargs)
        
        if not self.config['attack']['ulp']['DYNAMIC']:
            # non-adaptive ulp always start from a pretrained model
            self.config['network']['PRETRAINED'] = True
        
        models = NETWORK_BUILDER(config=self.config)
        models.build_network()
        
        # f_star
        self.model = models.model.module if self.config['train']['DISTRIBUTED'] else models.model
        self.dataset = databuilder.trainset
        
        self.normalizer = transforms.Normalize(
            mean = databuilder.mean, 
            std  = databuilder.std
        )
        
        # reset model to guarantee different initialization to avoid cheating
        if self.config['attack']['ulp']['DYNAMIC']:
            for model in self.model_list:
                for module in model.children():
                    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                        module.reset_parameters()


    def _add_trigger(self, 
                     img: np.ndarray, 
                     label: int) -> np.ndarray:
        
        return 0.5*img[None, :, :, :] + 0.5*(self.trigger[label].permute([0, 2, 3, 1]).detach().cpu().numpy())
    
    
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
        c, w, h =  images.shape[1], images.shape[2], images.shape[3]
        
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
                    images_k_perturb = (torch.clamp(0.5*images_k[:, None, :, :, :] + 0.5*self.uap[k][None, :, :, :, :], 0, 1)).view(-1, c, h, w)
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
    
    
class ULPATTACK(ATTACKER):
    
    def __init__(self, 
                 **kwargs) -> None:
        super().__init__(**kwargs)
        
        if not self.config['attack']['ulp']['DYNAMIC']:
            # non-adaptive ulp always start from a pretrained model
            self.config['network']['PRETRAINED'] = True
        
        self.model_list = []
        for i in range(self.config['attack']['ulp']['NUM_MODELS']):
            self.config['args']['checkpoint'] = self.config['attack']['ulp']['MODEL_POOL'][f'{self.argsnetwork}{self.argsdataset}'][i]
            models = NETWORK_BUILDER(config=self.config)
            models.build_network()
            model = models.model.module if self.config['train']['DISTRIBUTED'] else models.model
            if self.config['attack']['ulp']['DYNAMIC']:
                model.train()
            else:
                model.eval()
            self.model_list.append(model)
            
        self.dataset = DATA_BUILDER(self.config)
        self.dataset.build_dataset()
        
        self.normalizer = transforms.Normalize(
            mean = self.dataset.mean, 
            std = self.dataset.std
        )
        
        # reset model to guarantee different initialization to avoid cheating
        if self.config['attack']['ulp']['DYNAMIC']:
            for model in self.model_list:
                for module in model.children():
                    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                        module.reset_parameters()
        
        
    def _add_trigger(self, 
                     img: np.ndarray, 
                     label: int) -> np.ndarray:
        
        return (1-self.alpha)*img[None, :, :, :] + self.alpha*(self.trigger[label].detach().cpu().numpy())
    
    
    def _generate_trigger(self) -> np.ndarray:
        
        device = self.config['train']['device']
        self.ulp_dataset = self.dataset.trainset
        
        if self.config['attack']['ulp']['LABEL_CLEAN']:
            # clean label attack
            new_source_target_pair = dict()
            for _, v in self.config['attack']['SOURCE_TARGET_PAIR'].items():
                new_source_target_pair[v] = v
            self.target_source_pair = new_source_target_pair
        
        # choose data to inject trojan
        select_indices = [i for i in range(len(self.ulp_dataset.labels_c))]
        target_indices = [i for i in range(len(self.ulp_dataset.labels_c)) if int(self.ulp_dataset.labels_c[i]) in self.target_source_pair]
        select_indices = np.random.choice(target_indices, size=int(0.1*len(target_indices)), replace=False)
        
        c = self.config['dataset'][self.argsdataset]['NUM_CHANNELS']
        w, h = self.config['dataset'][self.argsdataset]['IMG_SIZE'], self.config['dataset'][self.argsdataset]['IMG_SIZE']
        # use to store UAP for each class. element is of shape N_ulp*C*H*W
        self.ulp = defaultdict(torch.Tensor)
        for k in self.target_source_pair:
            # initialize UAP
            self.ulp[int(k)] = torch.rand_like(torch.zeros([
                self.config['attack']['ulp']['N_ULP'],
                w, h, c
            ]), requires_grad=True)
        
        # self.ulp_dataset.select_data(np.array(select_indices))    
        batch_size  = self.config['train'][self.argsdataset]['BATCH_SIZE']//self.config['attack']['ulp']['N_ULP']
        dataloader  = torch.utils.data.DataLoader(self.ulp_dataset, batch_size=len(self.model_list)*int(self.config['attack']['ulp']['N_ULP'])*batch_size, shuffle=True)
        
        if not self.config['network']['PRETRAINED']:
            optimizer = torch.optim.SGD([{'params': model.parameters()} for model in self.model_list], 
                                        lr=float(self.config['train']['LR']), 
                                        weight_decay=float(self.config['train']['WEIGHT_DECAY']), 
                                        momentum=float(self.config['train']['MOMENTUM']), 
                                        nesterov=True)
        
        criterion_ce = torch.nn.CrossEntropyLoss()
        
        iters = 0
        foolrate = 0
        
        while (iters < self.config['attack']['ulp']['OPTIM_EPOCHS']):
            
            n_fooled = 0
            n_total  = 0
            
            for b, (indices, images, labels_c, labels_t) in enumerate(dataloader):
                
                images_perturb = []
                labels_clean = []
                labels_troj  = []
                
                for k in self.ulp:
                    troj_indices = [i for i in range(len(indices)) if indices[i].numpy() in np.intersect1d(indices[torch.where(labels_c == k)], select_indices)]
                    if len(troj_indices)!=0:
                        # add each ULP to each images
                        images_perturb.append((torch.clamp(0.5*images[troj_indices][:, None, :, :, :] + 0.5*self.ulp[k][None, :, :, :, :].permute(0, 1, 4, 2, 3), 0, 1)).view(-1, c, h, w))
                        labels_troj.append(torch.tensor(len(troj_indices)*len(self.ulp[k])*[int(self.target_source_pair[k])]))
                if len(images_perturb):
                    images_perturb = self.normalizer(torch.cat(images_perturb, 0))
                    labels_troj = torch.cat(labels_troj, 0)
                
                images_clean = self.normalizer(images)
                labels_clean = labels_c 
                
                batch_size_b = len(images_clean)//len(self.model_list)//int(self.config['attack']['ulp']['N_ULP'])
                
                loss_t = 0
                loss_c = 0
                # GPU memory saving purpose
                for i in range(len(self.model_list)):
                    
                    model_i = self.model_list[i].to(device)
                        
                    for j in range(int(self.config['attack']['ulp']['N_ULP'])):
                        
                        ind = i*int(self.config['attack']['ulp']['N_ULP']) + j
                        
                        images_clean_ij = images_clean[(ind*batch_size_b):(min((ind+1)*batch_size_b, len(images_clean)))].to(device)
                        labels_c_ij = labels_clean[(ind*batch_size_b):(min((ind+1)*batch_size_b, len(labels_clean)))].to(device)
                        outputs_c_ij = model_i(images_clean_ij)
                        loss_c += criterion_ce(outputs_c_ij, labels_c_ij)
                        
                        images_perturb_ij = images_perturb[(ind*batch_size_b):(min((ind+1)*batch_size_b, len(images_clean)))]
                        if len(images_perturb_ij):
                            images_perturb_ij = images_perturb_ij.to(device)
                            labels_t_ij = labels_troj[(ind*batch_size_b):(min((ind+1)*batch_size_b, len(labels_troj)))].to(device)
                            outputs_t_ij = model_i(images_perturb_ij)
                            loss_t += criterion_ce(outputs_t_ij, labels_t_ij)
                            
                            _, pred  = outputs_t_ij.max(1)
                            n_fooled = pred.eq(labels_t_ij).sum().item()
                            n_total  = len(labels_t_ij) 
                
                # gradient cumulate
                (loss_c/len(self.model_list)).backward()
                if len(images_perturb):
                    (loss_t/len(self.model_list)).backward()
                if b%self.config['train']['GRAD_CUMU_EPOCH']:
                    # ulp step
                    if self.ulp[k].grad is not None:
                        delta_ulp, self.ulp[k] = self.ulp[k].grad.data.detach(), self.ulp[k].detach()
                        self.ulp[k] -= delta_ulp
                        self.ulp[k] *= self.config['attack']['TRIGGER_SIZE']/(torch.norm(self.ulp[k], p=2)+1e-4)
                        self.ulp[k].requires_grad = True
                    
                if (not self.config['network']['PRETRAINED']) and (b%self.config['train']['GRAD_CUMU_EPOCH']):
                    for i in range(len(self.model_list)):
                        torch.nn.utils.clip_grad_norm_(self.model_list[i].parameters(), 100)
                    optimizer.step()
                    optimizer.zero_grad()
                    
            # import matplotlib.pyplot as plt
            # fig = plt.figure(figsize=(15, 5))
            # plt.subplot(1, 3, 1)
            # plt.imshow(images_k[0].detach().squeeze().permute([2,1,0]).cpu().numpy()/images_k[0].max().item())
            # plt.subplot(1, 3, 2)
            # plt.imshow(self.ulp[k].detach().squeeze().permute([2,1,0]).cpu().numpy()/self.ulp[k].max().item())
            # plt.subplot(1, 3, 3)
            # plt.imshow(images_k_perturb[0].squeeze().permute([2,1,0]).detach().cpu().numpy())
            # plt.savefig(f"./tmp/img_ulp_{iters}.png")
                    
            iters += 1
            foolrate = n_fooled/(n_total+1)
            print(f"[{iters:2d}|{self.config['attack']['ulp']['OPTIM_EPOCHS']:2d}] - Fooling Rate {foolrate:.3f} - {torch.norm(self.ulp[k], p=2):.3f}")
        self.trigger = self.ulp
        
        