from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from attacker import ATTACKER
from utils import AverageMeter

class TRAINER():

    def __init__(self, 
                 model: torch.nn.Module, 
                 attacker: ATTACKER, 
                 config: Dict, 
                 **kwargs) -> None:
        
        self.model = model
        self.config = config
        self.device = self.config['train']['device']
        self.model  = self.model.to(self.device) 
        self.attacker = attacker
        
        self.argsnetwork = config['args']['network']
        self.argsdataset = config['args']['dataset']
        self.argsmethod = config['args']['method']
        self.argsseed   = config['args']['seed']
                
        self.pretrained = config['network']['PRETRAINED']

        self.use_clip = config['train']['USE_CLIP']
        self.use_transform = config['train']['USE_TRANSFORM']
        self.advtrain = config['adversarial']['ADV_TRAIN']
        
        self.metric_history = {
            'train_ce_loss':     AverageMeter('train_ce_loss',   offset=1), 
            'train_clean_acc':   AverageMeter('train_clean_acc', offset=1), 
            'train_troj_acc' :   AverageMeter('train_troj_acc',  offset=1), 
            'train_overall_acc': AverageMeter('train_overall_acc', offset=1), 
            'test_ce_loss':      AverageMeter('test_ce_loss',  offset=1), 
            'test_clean_acc':    AverageMeter('test_clean_acc',offset=1), 
            'test_troj_acc' :    AverageMeter('test_troj_acc', offset=1), 
            'test_overall_acc':  AverageMeter('test_overall_acc', offset=1),   
        }
        
    def train(self, 
              trainloader: torch.utils.data.DataLoader, 
              validloader: torch.utils.data.DataLoader) -> None:
        
        self.trainloader = trainloader
        self.validloader = validloader
        
        if self.config['adversarial']['ADV_TRAIN']:
            self._adv_train()
        else:
            self._orig_train()
            

    def _orig_train(self) -> None:
        
        if self.config['train']['device'] == 0:
            self.timestamp = datetime.today().strftime('%y%m%d%H%M%S')
            self.logger = SummaryWriter(log_dir=self.config['args']['logdir'], 
                                        comment=self.argsdataset+'_'+self.argsnetwork+'_'+self.argsmethod+'_orig_'+self.timestamp, 
                                        flush_secs=30) 
        
        optimizer = torch.optim.SGD(self.model.parameters(), 
                                    lr=float(self.config['train']['LR']), 
                                    weight_decay=float(self.config['train']['WEIGHT_DECAY']), 
                                    momentum=float(self.config['train']['MOMENTUM']), 
                                    nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.config['train'][self.argsdataset]['T_MAX'])

        criterion_ce = torch.nn.CrossEntropyLoss().to(self.device)
        best_metric = 0

        for epoch in tqdm(range(int(self.config['train'][self.argsdataset]['N_EPOCHS'])), ncols=100, leave=True, position=0):
            
            for k in self.metric_history:
                self.metric_history[k].reset()
            
            if self.attacker.dynamic:
                self.attacker.reset_trojcount()
            
            if self.config['train']['DISTRIBUTED']:
                self.trainloader.sampler.set_epoch(epoch)
            
            self.model.train()
            for b, (ind, images, labels_c, labels_t) in enumerate(self.trainloader):
                
                if not b%2:
                    optimizer.zero_grad()
                
                if self.attacker.dynamic:
                    images_troj, labels_c2, labels_t2 = self.attacker.inject_trojan_dynamic(images, labels_c, epoch=epoch, batch=b, mode='train', backward=b%2)
                    if len(images_troj):
                        images   = torch.cat([images, images_troj], 0)
                        labels_c = torch.cat([labels_c, labels_c2], 0)
                        labels_t = torch.cat([labels_t, labels_t2], 0)
                
                images, labels_c, labels_t = images.to(self.device), labels_c.to(self.device), labels_t.to(self.device)
                
                outs = self.model(images)
                 
                loss = criterion_ce(outs, labels_t)
                loss.backward()
                
                if b%2:
                    optimizer.step()
                
                clean_ind  = torch.where(labels_c == labels_t)[0]
                troj_ind = torch.where(labels_c != labels_t)[0]
                
                _, pred = outs.max(1)
                    
                self.metric_history['train_ce_loss'].update(loss.item(), 1, epoch)
                self.metric_history['train_clean_acc'].update(pred[clean_ind].eq(labels_c[clean_ind]).sum().item(), len(clean_ind), epoch)
                self.metric_history['train_troj_acc'].update(pred[troj_ind].eq(labels_t[troj_ind]).sum().item(), len(troj_ind), epoch)
                self.metric_history['train_overall_acc'].update(pred.eq(labels_t).sum().item(), len(labels_t), epoch)
                
            scheduler.step()
    
            test_result = self.eval(self.validloader)
            for k in test_result:
                self.metric_history[k].update(test_result[k], 0, epoch)
            
            if (test_result['test_clean_acc']+test_result['test_troj_acc'])/2 > best_metric:
                self.best_model = self.model.module.state_dict()
    
            if self.config['train']['device'] == 0:
                
                self.logger.add_scalars(f"{self.argsnetwork}_{self.argsdataset}_{self.argsmethod}_{self.pretrained}_{self.use_clip}_{self.use_transform}_{self.advtrain}_{self.timestamp}_{self.argsseed}/Loss", {
                    'train': self.metric_history['train_ce_loss'].val, 
                    'test':  self.metric_history['test_ce_loss'].val
                    }, epoch)
                self.logger.add_scalars(f"{self.argsnetwork}_{self.argsdataset}_{self.argsmethod}_{self.pretrained}_{self.use_clip}_{self.use_transform}_{self.advtrain}_{self.timestamp}_{self.argsseed}/Overall_Acc", {
                    'train': self.metric_history['train_overall_acc'].val, 
                    'test':  self.metric_history['test_overall_acc'].val 
                    }, epoch)
                self.logger.add_scalars(f'{self.argsnetwork}_{self.argsdataset}_{self.argsmethod}_{self.pretrained}_{self.use_clip}_{self.use_transform}_{self.advtrain}_{self.timestamp}_{self.argsseed}/Clean_Acc', {
                    'train': self.metric_history['train_clean_acc'].val, 
                    'test':  self.metric_history['test_clean_acc'].val
                    }, epoch)
                self.logger.add_scalars(f'{self.argsnetwork}_{self.argsdataset}_{self.argsmethod}_{self.pretrained}_{self.use_clip}_{self.use_transform}_{self.advtrain}_{self.timestamp}_{self.argsseed}/Troj_Acc', {
                    'train': self.metric_history['train_troj_acc'].val, 
                    'test':  self.metric_history['test_troj_acc'].val
                    }, epoch)
                
                if bool(self.config['misc']['VERBOSE']) and (epoch%int(self.config['misc']['MONITOR_WINDOW'])==0):
                    tqdm.write(100*"-")
                    tqdm.write(f"[{epoch:2d}|{int(self.config['train'][self.argsdataset]['N_EPOCHS']):2d}] \t train loss:\t\t{self.metric_history['train_ce_loss'].val:.3f} \t\t train overall acc:\t{100*self.metric_history['train_overall_acc'].val:.3f}%")
                    tqdm.write(f"\t\t train clean acc:\t{100*self.metric_history['train_clean_acc'].val:.3f}% \t train troj acc:\t{100*self.metric_history['train_troj_acc'].val:.3f}%")
                    tqdm.write(f"\t\t test loss:\t\t{self.metric_history['test_ce_loss'].val:.3f} \t\t test overall acc:\t{100*self.metric_history['test_overall_acc'].val:.3f}%")
                    tqdm.write(f"\t\t test clean acc:\t{100*self.metric_history['test_clean_acc'].val:.3f}% \t test troj acc:\t\t{100*self.metric_history['test_troj_acc'].val:.3f}%")
        
        if hasattr(self, 'logger'):
            self.logger.close()
    
    
    def _adv_train(self) -> None:
        
        self.timestamp = datetime.today().strftime('%y%m%d%H%M%S')
        self.logger = SummaryWriter(log_dir=self.config['args']['logdir'], 
                            comment=self.config['args']['dataset']+'_'+self.config['args']['network']+'_'+self.config['args']['method']+'_adv_'+self.timestamp, 
                            flush_secs=30) 
        
        optimizer = torch.optim.SGD(self.model.parameters(), 
                                    lr=float(self.config['train']['LR']), 
                                    weight_decay=float(self.config['train']['WEIGHT_DECAY']), 
                                    momentum=float(self.config['train']['MOMENTUM']), 
                                    nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.config['train'][self.argsdataset]['T_MAX'])

        criterion_ce = torch.nn.CrossEntropyLoss().to(self.device)
        best_metric = 0

        # use free-m adversarial training
        for epoch in tqdm(range(int(self.config['train'][self.argsdataset]['N_EPOCHS'])//self.config['adversarial']['OPTIM_EPOCHS']), 
                          ncols=100, leave=True, position=0):
            
            for k in self.metric_history:
                if 'train' in k:
                    self.metric_history[k].reset()
            
            if self.config['train']['DISTRIBUTED']:
                self.trainloader.sampler.set_epoch(epoch)
            
            if self.attacker.dynamic:
                self.attacker.reset_trojcount()

            self.model.train()
            for b, (_, images, labels_c, labels_t) in enumerate(self.trainloader):
                
                if self.attacker.dynamic:
                    images_troj, labels_c2, labels_t2  = self.attacker.inject_trojan_dynamic(images, labels_c, epoch=epoch, batch=b, mode='train')
                    if len(images_troj):
                        delta_x_batch_troj = torch.zeros(images_troj.shape, dtype=images_troj.dtype).to(self.device)
                        
                        images = torch.cat([images, images_troj])
                        labels_c = torch.cat([labels_c, labels_c2])
                        labels_t = torch.cat([labels_t, labels_t2])
                        delta_x_batch = torch.cat([delta_x_batch, delta_x_batch_troj])
                
                images, labels_c, labels_t = images.to(self.device), labels_c.to(self.device), labels_t.to(self.device)
                delta_x_batch = torch.zeros(images.shape, dtype=images.dtype).to(self.device)
                
                for _ in range(int(self.config['adversarial']['OPTIM_EPOCHS'])):
                    
                    delta_x_batch.requires_grad = True
                    
                    outs_orig, outs_adv = self.model(images), self.model(images+delta_x_batch)
                    loss = criterion_ce(outs_orig, labels_t) + self.config['adversarial']['LAMBDA']*criterion_ce(outs_adv, labels_t)
                    optimizer.zero_grad()
                    loss.backward()
                    grad_delta_x_batch, delta_x_batch = delta_x_batch.grad.data.detach(), delta_x_batch.detach()
                    optimizer.step()
                    
                    delta_x_batch += float(self.config['adversarial']['EPS'])*grad_delta_x_batch
                    delta_x_batch_norm = torch.norm(delta_x_batch, p=2)
                    if delta_x_batch_norm > float(self.config['adversarial']['RADIUS']):
                        delta_x_batch = delta_x_batch/delta_x_batch_norm*float(self.config['adversarial']['RADIUS'])
                    
                clean_ind  = torch.where(labels_c == labels_t)[0]
                troj_ind = torch.where(labels_c != labels_t)[0]
                _, pred = outs_orig.max(1)
                
                self.metric_history['train_ce_loss'].update(loss.item(), 0, epoch)
                self.metric_history['train_clean_acc'].update(pred[clean_ind].eq(labels_c[clean_ind]).sum().item(), len(clean_ind), epoch)
                self.metric_history['train_troj_acc'].update(pred[troj_ind].eq(labels_t[troj_ind]).sum().item(), len(troj_ind), epoch)
                self.metric_history['train_overall_acc'].update(pred.eq(labels_t).sum().item(), len(labels_t), epoch)
                
            scheduler.step()
            
            if self.config['train']['device'] == 0:
                
                test_result = self.eval(self.validloader)
                for k in test_result:
                    self.metric_history[k].update(test_result[k], 1, epoch)

                if (test_result['test_clean_acc']+test_result['test_troj_acc'])/2 > best_metric:
                    self.best_model = self.model.module.state_dict()
                
                self.logger.add_scalars(f"{self.argsnetwork}_{self.argsdataset}_{self.argsmethod}_{self.pretrained}_{self.use_clip}_{self.use_transform}_{self.advtrain}_{self.timestamp}_{self.argsseed}/Loss", {
                    'train': self.metric_history['train_ce_loss'].val, 
                    'test':  self.metric_history['test_ce_loss'].val
                    }, epoch)
                self.logger.add_scalars(f"{self.argsnetwork}_{self.argsdataset}_{self.argsmethod}_{self.pretrained}_{self.use_clip}_{self.use_transform}_{self.advtrain}_{self.timestamp}_{self.argsseed}/Overall_Acc", {
                    'train': self.metric_history['train_overall_acc'].val, 
                    'test':  self.metric_history['test_overall_acc'].val 
                    }, epoch)
                self.logger.add_scalars(f'{self.argsnetwork}_{self.argsdataset}_{self.argsmethod}_{self.pretrained}_{self.use_clip}_{self.use_transform}_{self.advtrain}_{self.timestamp}_{self.argsseed}/Clean_Acc', {
                    'train': self.metric_history['train_clean_acc'].val, 
                    'test':  self.metric_history['test_clean_acc'].val
                    }, epoch)
                self.logger.add_scalars(f'{self.argsnetwork}_{self.argsdataset}_{self.argsmethod}_{self.pretrained}_{self.use_clip}_{self.use_transform}_{self.advtrain}_{self.timestamp}_{self.argsseed}/Troj_Acc', {
                    'train': self.metric_history['train_troj_acc'].val, 
                    'test':  self.metric_history['test_troj_acc'].val
                    }, epoch)
                
                if bool(self.config['misc']['VERBOSE']) and (epoch%int(self.config['misc']['MONITOR_WINDOW'])==0):
                    tqdm.write(100*"-")
                    tqdm.write(f"[{epoch:2d}|{int(self.config['train'][self.argsdataset]['N_EPOCHS']):2d}] \t train loss:\t\t{self.metric_history['train_ce_loss'].val:.3f} \t\t train overall acc:\t{100*self.metric_history['train_overall_acc'].val:.3f}%")
                    tqdm.write(f"\t\t train clean acc:\t{100*self.metric_history['train_clean_acc'].val:.3f}% \t train troj acc:\t{100*self.metric_history['train_troj_acc'].val:.3f}%")
                    tqdm.write(f"\t\t test loss:\t\t{self.metric_history['test_ce_loss'].val:.3f} \t\t test overall acc:\t{100*self.metric_history['test_overall_acc'].val:.3f}%")
                    tqdm.write(f"\t\t test clean acc:\t{100*self.metric_history['test_clean_acc'].val:.3f}% \t test troj acc:\t\t{100*self.metric_history['test_troj_acc'].val:.3f}%")

    
        self.logger.close()
        
    
    def eval(self, evalloader: torch.utils.data.DataLoader, load_checkpoint: bool=False) -> Dict:
        
        if load_checkpoint:
            if self.config['train']['DISTRIBUTED']:
                self.model.module.load_state_dict(self.best_model)
            else:
                self.model.load_state_dict(self.best_model)
        
        criterion_ce = torch.nn.CrossEntropyLoss()

        ce_loss = AverageMeter('test_ce_loss', offset=1)
        troj_acc  = AverageMeter('test_troj_acc',  offset=1)
        clean_acc = AverageMeter('test_clean_acc', offset=1)
        overall_acc = AverageMeter('test_overall_acc', offset=1)
        
        self.model.eval()
        for b, (_, images, labels_c, labels_t) in enumerate(evalloader):
            
            if self.attacker.dynamic: 
                self.attacker.reset_trojcount()
                
                iamges_troj, labels_c2, labels_t2 = self.attacker.inject_trojan_dynamic(images, labels_c, mode='test')
                if len(iamges_troj):
                    images = torch.cat([images, iamges_troj], 0)
                    labels_c = torch.cat([labels_c, labels_c2])
                    labels_t = torch.cat([labels_t, labels_t2])
            
            images, labels_c, labels_t = images.to(self.device), labels_c.to(self.device), labels_t.to(self.device)
            
            if self.config['train']['DISTRIBUTED']:
                outs = self.model.module(images)
            else:
                outs = self.model(images)
            loss = criterion_ce(outs, labels_t)

            clean_ind  = torch.where(labels_c == labels_t)[0]
            troj_ind = torch.where(labels_c != labels_t)[0]
            
            _, pred = outs.max(1)
            
            ce_loss.update(loss.item(), 1, 0)
            clean_acc.update(pred[clean_ind].eq(labels_c[clean_ind]).sum().item(), len(clean_ind), 0)
            troj_acc.update(pred[troj_ind].eq(labels_t[troj_ind]).sum().item(), len(troj_ind), 0)
            overall_acc.update(pred.eq(labels_t).sum().item(), len(labels_t), 0)

        return {
                'test_ce_loss': ce_loss.val, 
                'test_clean_acc': clean_acc.val, 
                'test_troj_acc': troj_acc.val, 
                'test_overall_acc': overall_acc.val
                }
