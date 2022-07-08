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
        
        self.network = config['args']['network']
        self.dataset = config['args']['dataset']
        self.method = config['args']['method']
        self.pretrained = config['network']['PRETRAINED']
        self.advtrain = config['adversarial']['ADV_TRAIN']
        
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
        
        self.timestamp = datetime.today().strftime('%y%m%d%H%M%S')
        self.logger = SummaryWriter(log_dir=self.config['args']['logdir'], 
                            comment=self.config['args']['dataset']+'_'+self.config['args']['network']+'_'+self.config['args']['method']+'_orig_'+self.timestamp, 
                            flush_secs=30) 
        
        optimizer = torch.optim.SGD(self.model.parameters(), 
                                    lr=float(self.config['train']['LR']), 
                                    weight_decay=float(self.config['train']['WEIGHT_DECAY']), 
                                    momentum=float(self.config['train']['MOMENTUM']), 
                                    nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                                               T_max=self.config['train'][self.dataset]['T_MAX'])

        criterion_ce = torch.nn.CrossEntropyLoss()

        ce_loss = AverageMeter('ce_loss')
        clean_acc = AverageMeter('clean_acc')
        troj_acc  = AverageMeter('troj_acc')
        overall_acc = AverageMeter('overall_acc')

        for epoch in tqdm(range(int(self.config['train'][self.dataset]['N_EPOCHS'])), ncols=100, leave=True, position=0):
            
            ce_loss.reset()
            clean_acc.reset()
            troj_acc.reset()
            overall_acc.reset()

            self.model.train()
            for _, (ind, images, labels_c, labels_t) in enumerate(self.trainloader):
                images, labels_c, labels_t = images.to(self.device), labels_c.to(self.device), labels_t.to(self.device)
                outs = self.model(images)
                loss = criterion_ce(outs, labels_t)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                clean_ind  = torch.where(labels_c == labels_t)[0]
                troj_ind = torch.where(labels_c != labels_t)[0]
                
                _, pred = outs.max(1)
                ce_loss.update(loss, len(labels_c))
                clean_acc.update(pred[clean_ind].eq(labels_c[clean_ind]).sum().item(), len(clean_ind))
                troj_acc.update(pred[troj_ind].eq(labels_t[troj_ind]).sum().item(), len(troj_ind))
                overall_acc.update(pred.eq(labels_t).sum().item(), len(labels_t))
                
                if self.attacker.dynamic:
                    images, labels_c, labels_t = self.attacker.inject_trojan_dynamic(images, labels_c)
                    outs_troj = self.model(images)
                    loss = criterion_ce(outs_troj, labels_t)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    clean_ind  = torch.where(labels_c == labels_t)[0]
                    troj_ind = torch.where(labels_c != labels_t)[0]
                
                    _, pred = outs.max(1)
                    ce_loss.update(loss, len(labels_c))
                    clean_acc.update(pred[clean_ind].eq(labels_c[clean_ind]).sum().item(), len(clean_ind))
                    troj_acc.update(pred[troj_ind].eq(labels_t[troj_ind]).sum().item(), len(troj_ind))
                    overall_acc.update(pred.eq(labels_t).sum().item(), len(labels_t))

            scheduler.step()

            test_result = self.eval(self.validloader)
            
            self.logger.add_scalars(f"{self.network}_{self.dataset}_{self.method}_{self.pretrained}_{self.advtrain}_{self.timestamp}/Loss", {
                'train': ce_loss.val, 
                'test': test_result['ce_loss'].val
                }, epoch)
            self.logger.add_scalars(f"{self.network}_{self.dataset}_{self.method}_{self.pretrained}_{self.advtrain}_{self.timestamp}/Overall_Acc", {
                'train': overall_acc.val, 
                'test': test_result['overall_acc'].val
                }, epoch)
            self.logger.add_scalars(f'{self.network}_{self.dataset}_{self.method}_{self.pretrained}_{self.advtrain}_{self.timestamp}/Clean_Acc', {
                'train': clean_acc.val, 
                'test': test_result['clean_acc'].val
                }, epoch)
            self.logger.add_scalars(f'{self.network}_{self.dataset}_{self.method}_{self.pretrained}_{self.advtrain}_{self.timestamp}/Troj_Acc', {
                'train': troj_acc.val, 
                'test': test_result['troj_acc'].val
                }, epoch)
            
            if bool(self.config['misc']['VERBOSE']) and (epoch%int(self.config['misc']['MONITOR_WINDOW'])==0):
                tqdm.write(100*"-")
                tqdm.write(f"[{epoch:2d}|{int(self.config['train'][self.dataset]['N_EPOCHS']):2d}] \t train loss:\t\t{ce_loss.val:.3f} \t\t train overall acc:\t{overall_acc.val*100:.3f}%")
                tqdm.write(f"\t\t train clean acc:\t{clean_acc.val*100:.3f}% \t train troj acc:\t{troj_acc.val*100:.3f}%")
                tqdm.write(f"\t\t test loss:\t\t{test_result['ce_loss'].val:.3f} \t\t test overall acc:\t{test_result['overall_acc'].val*100:.3f}%")
                tqdm.write(f"\t\t test clean acc:\t{test_result['clean_acc'].val*100:.3f}% \t test troj acc:\t\t{test_result['troj_acc'].val*100:.3f}%")
                
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                                               T_max=self.config['train'][self.dataset]['T_MAX'])

        criterion_ce = torch.nn.CrossEntropyLoss()

        ce_loss = AverageMeter('ce_loss')
        clean_acc = AverageMeter('clean_acc')
        troj_acc  = AverageMeter('troj_acc')
        overall_acc = AverageMeter('overall_acc')

        # use free-m adversarial training
        for epoch in tqdm(range(int(self.config['train'][self.dataset]['N_EPOCHS'])//self.config['adversarial']['OPTIM_EPOCHS']), 
                          ncols=100, 
                          leave=True, 
                          position=0):
            
            ce_loss.reset()
            clean_acc.reset()
            troj_acc.reset()
            overall_acc.reset()

            self.model.train()
            for _, (_, images, labels_c, labels_t) in enumerate(self.trainloader):
                
                delta_x_batch = torch.zeros(images.shape, dtype=images.dtype).to(self.device)
                
                if self.attacker.dynamic:
                    images_troj, labels_clean, labels_troj = self.attacker.inject_trojan_dynamic(images, labels_c)
                    delta_x_batch_troj = torch.zeros(images_troj.shape, dtype=images_troj.dtype).to(self.device)
                
                for _ in range(int(self.config['adversarial']['OPTIM_EPOCHS'])):
                    
                    delta_x_batch.requires_grad = True
                    
                    images, labels_c, labels_t = images.to(self.device), labels_c.to(self.device), labels_t.to(self.device)
                    outs_orig, outs_adv = self.model(images), self.model(images+delta_x_batch)
                    loss = criterion_ce(outs_orig, labels_t) + self.config['adversarial']['LAMBDA']*criterion_ce(outs_adv, labels_t)
                    optimizer.zero_grad()
                    loss.backward()
                    grad_delta_x_batch, delta_x_batch = delta_x_batch.grad.data.detach(), delta_x_batch.detach()
                    optimizer.step()
                    
                    if self.attacker.dynamic:
                        outs_troj, outs_adv = self.model(images_troj), self.model(images_troj+delta_x_batch_troj)
                        loss = criterion_ce(outs_troj, labels_troj) + self.config['adversarial']['LAMBDA']*criterion_ce(outs_adv, labels_troj)
                        optimizer.zero_grad()
                        loss.backward()
                        grad_delta_x_batch_troj, delta_x_batch_troj = delta_x_batch_troj.grad.data.detach(), delta_x_batch_troj.detach()
                        optimizer.step()
                    
                    delta_x_batch += float(self.config['adversarial']['EPS'])*grad_delta_x_batch
                    delta_x_batch_norm = torch.norm(delta_x_batch, p=2)
                    if delta_x_batch_norm > float(self.config['adversarial']['RADIUS']):
                        delta_x_batch = delta_x_batch/delta_x_batch_norm*float(self.config['adversarial']['RADIUS'])
                    
                    if self.attacker.dynamic:
                        delta_x_batch_troj += float(self.config['adversarial']['EPS'])*grad_delta_x_batch_troj
                        delta_x_batch_norm_troj = torch.norm(delta_x_batch_troj, p=2)
                        if delta_x_batch_norm_troj > float(self.config['adversarial']['RADIUS']):
                            delta_x_batch_troj = delta_x_batch_troj/delta_x_batch_norm_troj*float(self.config['adversarial']['RADIUS'])

                clean_ind  = torch.where(labels_c == labels_t)[0]
                troj_ind = torch.where(labels_c != labels_t)[0]
                _, pred = outs_orig.max(1)
                ce_loss.update(loss, len(labels_c))
                clean_acc.update(pred[clean_ind].eq(labels_c[clean_ind]).sum().item(), len(clean_ind))
                troj_acc.update(pred[troj_ind].eq(labels_t[troj_ind]).sum().item(), len(troj_ind))
                overall_acc.update(pred.eq(labels_t).sum().item(), len(labels_t))
                
                if self.attacker.dynamic:
                    clean_ind = torch.where(labels_clean == labels_troj)[0]
                    troj_ind  = torch.where(labels_clean != labels_troj)[0]
                    _, pred = outs_troj.max(1)
                    clean_acc.update(pred[clean_ind].eq(labels_clean[clean_ind]).sum().item(), len(clean_ind))
                    troj_acc.update(pred[troj_ind].eq(labels_troj[troj_ind]).sum().item(), len(troj_ind))
                    overall_acc.update(pred.eq(labels_troj).sum().item(), len(labels_troj))
                    
            scheduler.step()
            
            test_result = self.eval(self.validloader)
            self.logger.add_scalars(f"{self.network}_{self.dataset}_{self.method}_{self.pretrained}_{self.advtrain}_{self.timestamp}/Loss", {
                'train': ce_loss.val, 
                'test': test_result['ce_loss'].val
                }, epoch)
            self.logger.add_scalars(f"{self.network}_{self.dataset}_{self.method}_{self.pretrained}_{self.advtrain}_{self.timestamp}/Overall_Acc", {
                'train': overall_acc.val, 
                'test': test_result['overall_acc'].val
                }, epoch)
            self.logger.add_scalars(f'{self.network}_{self.dataset}_{self.method}_{self.pretrained}_{self.advtrain}_{self.timestamp}/Clean_Acc', {
                'train': clean_acc.val, 
                'test': test_result['clean_acc'].val
                }, epoch)
            self.logger.add_scalars(f'{self.network}_{self.dataset}_{self.method}_{self.pretrained}_{self.advtrain}_{self.timestamp}/Troj_Acc', {
                'train': troj_acc.val, 
                'test': test_result['troj_acc'].val
                }, epoch)
            
            if bool(self.config['misc']['VERBOSE']) and (epoch%int(self.config['misc']['MONITOR_WINDOW'])==0):
                tqdm.write(100*"-")
                tqdm.write(f"[{epoch:2d}|{int(self.config['train'][self.dataset]['N_EPOCHS'])//self.config['adversarial']['OPTIM_EPOCHS']:2d}] \t train loss:\t\t{ce_loss.val:.3f} \t\t train overall acc:\t{overall_acc.val*100:.3f}%")
                tqdm.write(f"\t\t train clean acc:\t{clean_acc.val*100:.3f}% \t train troj acc:\t{troj_acc.val*100:.3f}%")
                tqdm.write(f"\t\t test loss:\t\t{test_result['ce_loss'].val:.3f} \t\t test overall acc:\t{test_result['overall_acc'].val*100:.3f}%")
                tqdm.write(f"\t\t test clean acc:\t{test_result['clean_acc'].val*100:.3f}% \t test troj acc:\t\t{test_result['troj_acc'].val*100:.3f}%")
    
        self.logger.close()
        
    
    def eval(self, evalloader: torch.utils.data.DataLoader) -> Dict:
        
        ce_loss = AverageMeter('ce_loss')
        clean_acc = AverageMeter('clean_acc')
        troj_acc  = AverageMeter('troj_acc')
        overall_acc = AverageMeter('overall_acc')

        criterion_ce = torch.nn.CrossEntropyLoss()

        self.model.eval()
        for _, (_, images, labels_c, labels_t) in enumerate(evalloader):
            images, labels_c, labels_t = images.to(self.device), labels_c.to(self.device), labels_t.to(self.device)
            outs = self.model(images)
            loss = criterion_ce(outs, labels_t)

            clean_ind  = torch.where(labels_c == labels_t)[0]
            troj_ind = torch.where(labels_c != labels_t)[0]
            
            _, pred = outs.max(1)
            ce_loss.update(loss.item(), len(labels_c))
            clean_acc.update(pred[clean_ind].eq(labels_c[clean_ind]).sum().item(), len(clean_ind))
            troj_acc.update(pred[troj_ind].eq(labels_t[troj_ind]).sum().item(), len(troj_ind))
            overall_acc.update(pred.eq(labels_t).sum().item(), len(labels_t))
            
            if self.attacker.dynamic:
                images_troj, labels_clean, labels_troj = self.attacker.inject_trojan_dynamic(images)
                outs_troj = self.model(images_troj)
                loss = criterion_ce(outs_troj, labels_troj)
                
                clean_ind = torch.where(labels_clean == labels_troj)[0]
                troj_ind  = torch.where(labels_clean != labels_troj)[0]
                
                _, pred = outs_troj.max(1)
                ce_loss.update(loss.item(), len(labels_clean))
                clean_acc.update(pred[clean_ind].eq(labels_clean[clean_ind]).sum().item(), len(clean_ind))
                troj_acc.update(pred[troj_ind].eq(labels_troj[troj_ind]).sum().item(), len(troj_ind))
                overall_acc.update(pred.eq(labels_troj).sum().item(), len(labels_troj))

        return {
                'ce_loss': ce_loss, 
                'clean_acc': clean_acc, 
                'troj_acc': troj_acc, 
                'overall_acc': overall_acc
                }
