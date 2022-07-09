from typing import List, Dict

import torch

class AverageMeter():

    def __init__(self, name:str) -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.count = 0
        self.total = 0

    def update(self, count: int, total: int) -> None:
        self.count += count 
        self.total += total 
        self.val = self.count/(self.total+1)
        
        
class DENORMALIZER: 
    
    def __init__(self,  mean: List, std: List, config: Dict, **kwargs) -> None:
        self.n_channels = config['dataset'][config['args']['dataset']]['NUM_CHANNELS']
        self.mean = mean 
        self.std = std
        
        assert self.n_channels == len(self.mean), f"Expect to have {self.n_channels} channels but got {len(self.mean)} !"
        
    
    def __call__(self, x: torch.tensor) -> torch.tensor:
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel, :] = self.std[channel]*x[:, channel, :] + self.mean[channel]
        return x_clone