from typing import Dict

import torch
from tqdm import tqdm

from utils import AverageMeter

class TRAINER():

    def __init__(self, 
                 model: torch.nn.Module, 
                 config: Dict, 
                 **kwargs) -> None:
        
        self.model = model
        self.config = config
        self.device = self.config['train']['device']
        self.model = self.model.to(self.device)

    def train(self, 
              trainloader: torch.utils.data.DataLoader, 
              validloader: torch.utils.data.DataLoader) -> None:
        
        optimizer = torch.optim.SGD(self.model.parameters(), 
                                    lr=float(self.config['train']['LR']), 
                                    weight_decay=float(self.config['train']['WEIGHT_DECAY']), 
                                    momentum=float(self.config['train']['MOMENTUM']), 
                                    nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                                               T_max=self.config['train']['cifar10']['T_MAX'])

        criterion_ce = torch.nn.CrossEntropyLoss()

        ce_loss = AverageMeter('ce_loss')
        clean_acc = AverageMeter('clean_acc')
        troj_acc  = AverageMeter('troj_acc')
        overall_acc = AverageMeter('overall_acc')

        for epoch in tqdm(range(int(self.config['train']['cifar10']['N_EPOCHS'])), ncols=100, leave=True, position=0):
            
            ce_loss.reset()
            clean_acc.reset()
            troj_acc.reset()
            overall_acc.reset()

            self.model.train()
            for _, (_, images, labels_c, labels_t) in enumerate(trainloader):
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

            scheduler.step()

            #TODO: add logger 

            if bool(self.config['misc']['VERBOSE']) and (epoch%int(self.config['misc']['MONITOR_WINDOW'])==0):
                test_result = self.eval(validloader)
                tqdm.write(100*"-")
                tqdm.write(f"[{epoch:2d}|{int(self.config['train']['cifar10']['N_EPOCHS']):2d}] \t train loss:\t\t{ce_loss.val:.3f} \t\t train overall acc:\t{overall_acc.val*100:.3f}%")
                tqdm.write(f"\t\t train clean acc:\t{clean_acc.val*100:.3f}% \t train troj acc:\t{troj_acc.val*100:.3f}%")
                tqdm.write(f"\t\t test loss:\t\t{test_result['ce_loss'].val:.3f} \t\t test overall acc:\t{test_result['overall_acc'].val*100:.3f}%")
                tqdm.write(f"\t\t test clean acc:\t{test_result['clean_acc'].val*100:.3f}% \t test troj acc:\t\t{test_result['troj_acc'].val*100:.3f}%")
            
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

        return {
                'ce_loss': ce_loss, 
                'clean_acc': clean_acc, 
                'troj_acc': troj_acc, 
                'overall_acc': overall_acc
                }

