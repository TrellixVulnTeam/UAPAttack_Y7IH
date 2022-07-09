import os
import os.path
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import csv
import pathlib
from typing import Any, Callable, Optional, Tuple, List
import random
from collections import defaultdict
sys.path.append("./")
sys.path.append("../")

import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as VF
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset
import torchvision.models as models
import numpy as np
from datetime import datetime
import pickle as pkl
import yaml
import PIL

from trainer import TRAINER
from networks import ResNet18, VGG16, DenseNet121


class GTSRB(VisionDataset):
    """`German Traffic Sign Recognition Benchmark (GTSRB) <https://benchmark.ini.rub.de/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "gtsrb"
        self._target_folder = (
            self._base_folder / "GTSRB" / ("Training" if self._split == "train" else "Final_Test/Images")
        )

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if self._split == "train":
            samples = make_dataset(str(self._target_folder), extensions=(".ppm",))
        else:
            with open(self._base_folder / "GT-final_test.csv") as csv_file:
                samples = [
                    (str(self._target_folder / row["Filename"]), int(row["ClassId"]))
                    for row in csv.DictReader(csv_file, delimiter=";", skipinitialspace=True)
                ]

        self._samples = samples

        self.troj_data = []
        self.troj_labels_c = []
        self.troj_labels_t = []
        
        self.transform = transform
        self.target_transform = target_transform

        self.clean_num = len(self._samples)
        
        self.use_transform = True

    def __len__(self) -> int:
        return len(self._samples)+len(self.troj_data)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if index < len(self._samples):
            path, target_c = self._samples[index]
            target_t = target_c
            sample = PIL.Image.open(path).convert("RGB")
        else:
            sample = self.troj_data[index-self.clean_num]
            target_c, target_t = self.troj_labels_c[index-self.clean_num], self.troj_labels_t[index-self.clean_num]
        
        if self.use_transform and self.transform is not None:
            sample = self.transform(sample)
        else:
            sample = VF.resize(VF.to_tensor(sample), (32, 32))

        if self.target_transform is not None:
            target_c = self.target_transform(target_c)
            target_t = self.target_transform(target_t)
        else:
            target_c = torch.tensor(target_c)
            target_t = torch.tensor(target_t) 
            
        return index, sample.float(), torch.tensor(target_c), torch.tensor(target_t)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


    def _check_exists(self) -> bool:
        return self._target_folder.is_dir()


    def download(self) -> None:
        if self._check_exists():
            return

        base_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"

        if self._split == "train":
            download_and_extract_archive(
                f"{base_url}GTSRB-Training_fixed.zip",
                download_root=str(self._base_folder),
                md5="513f3c79a4c5141765e10e952eaa2478",
            )
        else:
            download_and_extract_archive(
                f"{base_url}GTSRB_Final_Test_Images.zip",
                download_root=str(self._base_folder),
                md5="c7e4e6327067d32654124b0fe9e82185",
            )
            download_and_extract_archive(
                f"{base_url}GTSRB_Final_Test_GT.zip",
                download_root=str(self._base_folder),
                md5="fe31e9c9270bbcd7b84b7f21a9d9d9e5",
            )


    def insert_data(self, new_data: List, new_labels_c: np.ndarray, new_labels_t: np.ndarray) -> None:
        
        assert isinstance(new_data, list), "data need to be a list, but find " + str(type(new_data)) 
        assert isinstance(new_labels_c, np.ndarray), f"labels need to be a np.ndarray, but find " + str(type(new_labels_c))
        assert isinstance(new_labels_t, np.ndarray), f"labels need to be a np.ndarray, but find " + str(type(new_labels_t))

        self.troj_data += new_data
        self.troj_labels_c = torch.cat([torch.tensor(self.troj_labels_c), torch.tensor(new_labels_c)]).long()
        self.troj_labels_t = torch.cat([torch.tensor(self.troj_labels_t), torch.tensor(new_labels_t)]).long()
        
    def select_data(self, indices: np.ndarray) -> None:
        self._samples = [self._samples[i] for i in indices]
    
    
if __name__ == '__main__':
    
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('experiment_configuration.yml', 'r') as f:
        config = yaml.safe_load(f)
    f.close()
    config['train']['device'] = device
    config['train']['gtsrb']['N_EPOCHS'] = 100
    config['args'] = defaultdict(str)
    config['args']['dataset'] = 'gtsrb'
    config['args']['network'] = 'densenet121'
    config['args']['method'] = 'clean'

    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
    ])

    trainset = GTSRB(root="./data", split='train', transform=transform_train, download=True)
    testset  = GTSRB(root="./data", split='test',  transform=transform_test, download=True)
    trainloader = DataLoader(trainset, batch_size=int(config['train']['gtsrb']['BATCH_SIZE']), shuffle=True, pin_memory=True, num_workers=1)
    testloader  = DataLoader(testset, batch_size=int(config['train']['gtsrb']['BATCH_SIZE']))

    # For resnet18
    # model = ResNet18(num_classes=43).to(device)
    # model = VGG16(num_classes=43).to(device)
    model = DenseNet121(num_classes=43).to(device)
    
    model_trainer = TRAINER(model=model, config=config)
    model_trainer.train(trainloader, testloader)

    result_dict = model_trainer.eval(testloader)
    result_dict['model_state_dict'] = model_trainer.model.state_dict()
    result_dict['config'] = config

    time_stamp = datetime.today().strftime("%Y%m%d%H%M%S")
    result_file = f"clean_models/{config['args']['dataset']}_{config['args']['network']}_{config['args']['method']}_{time_stamp}.pkl"
    with open(result_file, 'wb') as f:
        pkl.dump(result_dict, f)
    f.close()
