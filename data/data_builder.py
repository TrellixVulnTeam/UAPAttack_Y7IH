from typing import Dict

from torch.utils.data import DataLoader
from torchvision import transforms

from .CIFAR import CIFAR10
from .GTSRB import GTSRB
from .IMAGENET import ImageNet

class DATA_BUILDER():

    def __init__(self, 
                 config: Dict) -> None:
        self.config = config
    
    def build_dataset(self) -> None:

        if self.config['args']['dataset'] == 'cifar10':
            self.num_classes = self.config['dataset']['cifar10']['NUM_CLASSES']
            
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std  = (0.2023, 0.1994, 0.2010)
            
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

            self.trainset = CIFAR10(root="./data", split='train', transform=transform_train, train_ratio=1, download=True)
            self.testset  = CIFAR10(root="./data", split='test',  transform=transform_test, download=True)
            self.trainloader = DataLoader(self.trainset, batch_size=self.config['train']['cifar10']['BATCH_SIZE'], shuffle=True, drop_last=True)
            self.testloader  = DataLoader(self.testset, batch_size=self.config['train']['cifar10']['BATCH_SIZE'], drop_last=True)
        
        elif self.config['args']['dataset'] == 'gtsrb':
            self.num_classes = self.config['dataset']['gtsrb']['NUM_CLASSES']
            
            self.mean = (0.3337, 0.3064, 0.3171)
            self.std  = (0.2672, 0.2564, 0.2629)
            
            transform_train = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

            transform_test = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])    
            
            self.trainset = GTSRB(root="./data", split='train', transform=transform_train, download=True)
            self.testset  = GTSRB(root="./data", split='test',  transform=transform_test, download=True)
            self.trainloader = DataLoader(self.trainset, batch_size=self.config['train']['gtsrb']['BATCH_SIZE'], shuffle=True, drop_last=True)
            self.testloader  = DataLoader(self.testset, batch_size=self.config['train']['gtsrb']['BATCH_SIZE'], drop_last=True)
        
        elif self.config['args']['dataset'] == 'imagenet':
            self.num_classes = self.config['dataset']['imagenet']['NUM_CLASSES']
            
            self.mean = (0.485, 0.456, 0.406)
            self.std  = (0.229, 0.224, 0.225)
            
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            self.trainset = ImageNet(root='/scr/songzhu/imagenet', split='train', transform=train_transform)
            self.testset  = ImageNet(root='/scr/songzhu/imagenet', split='val', transform=test_transform)
            self.trainloader = DataLoader(self.trainset, batch_size=self.config['train']['imagenet']['BATCH_SIZE'], shuffle=True, num_workers=1, drop_last=True)
            self.testloader  = DataLoader(self.testset, batch_size=self.config['train']['imagenet']['BATCH_SIZE'], num_workers=1, drop_last=True)
        
        else:
            raise NotImplementedError