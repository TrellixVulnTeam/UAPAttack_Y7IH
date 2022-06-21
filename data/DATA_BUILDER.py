from typing import Dict

from torch.utils.data import DataLoader
from torchvision import transforms

from CIFAR import CIFAR10

class DATA_BUILDER():

    def __init__(self, 
                 config: Dict) -> None:
        self.config = config
    
    def build_dataset(self) -> None:

        if self.config['args']['dataset'] == 'cifar10':
            self.num_classes = 10

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            self.trainset = CIFAR10(root="./data", split='train', transform=transform_train, train_ratio=1, download=True)
            self.testset  = CIFAR10(root="./data", split='test',  transform=transform_test, download=True)
            self.trainloader = DataLoader(self.trainset, batch_size=self.config['train']['BATCH_SIZE'], shuffle=True, pin_memory=True, num_workers=1)
            self.testloader  = DataLoader(self.testset, batch_size=self.config['train']['BATCH_SIZE'])
        else:
            raise NotImplementedError