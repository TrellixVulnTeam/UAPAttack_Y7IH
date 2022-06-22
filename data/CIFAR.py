import os
import os.path
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from typing import List
import random
sys.path.append("../")
sys.path.append("/home/songzhu/PycharmProjects/UAPAttack")

import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import yaml
import pickle as pkl
from datetime import datetime

from data.data_utils import download_url, check_integrity
from TRAINER import TRAINER
from networks import ResNet18


class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, 
                 root, 
                 split='train', 
                 train_ratio=0.8,
                 transform=None, 
                 target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set, validation set or test set
        self.train_ratio = train_ratio

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.split == 'test':
            downloaded_list = self.test_list
        else:
            downloaded_list = self.train_list

        self.data = []
        self.labels_c = []
        self.labels_t = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.labels_c.extend(entry['labels'])
                    self.labels_t.extend(entry['labels'])
                else:
                    self.labels_c.extend(entry['fine_labels'])
                    self.labels_t.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        # split the original train set into train & validation set
        if self.split != 'test':
            num_data = len(self.data)
            self.num_class = len(np.unique(self.labels_c))
            train_num = int(num_data * self.train_ratio)

            if self.split == 'train':
                self.data = self.data[:train_num]
                self.labels_c = self.labels_c[:train_num]
                self.labels_t = self.labels_t[:train_num]
            else: #valid
                self.data = self.data[train_num:num_data]
                self.labels_c = self.labels_c[train_num:num_data]
                self.labels_t = self.labels_t[train_num:num_data]
        else:
            num_data = len(self.data)
            self.num_class = len(np.unique(self.labels_c))

        self.labels_c = torch.tensor(self.labels_c)
        self.labels_t = torch.tensor(self.labels_t)
        self._load_meta()

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def demean(self):
        m = self.data.mean((0, 1))
        for i in range(self.data.shape[0]):
            self.data[i, 0] = self.data[i, 0] - m
            self.data[i, 1] = self.data[i, 1] - m
            self.data[i, 2] = self.data[i, 2] - m

    def insert_data(self, new_data: np.ndarray, new_labels_c: np.ndarray, new_labels_t: np.ndarray) -> None:
        assert isinstance(new_data, np.ndarray), "data need to be a np.ndarray, but find " + str(type(new_data)) 
        assert isinstance(new_labels_c, np.ndarray), f"labels need to be a np.ndarray, but find " + str(type(new_labels_c))
        assert isinstance(new_labels_t, np.ndarray), f"labels need to be a np.ndarray, but find " + str(type(new_labels_t))
        self.data = np.concatenate([self.data, new_data], 0)
        self.labels_c = np.concatenate([self.labels_c, new_labels_c], 0)
        self.labels_t = np.concatenate([self.labels_t, new_labels_t], 0)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, labels_c, labels_t = self.data[index], self.labels_c[index], self.labels_t[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            labels_c = self.target_transform(labels_c)
            labels_t = self.target_transform(labels_t)

        return index, img, labels_c, labels_t

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


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

# TODO: remove upon releasing
# generate clean model for this dataset 
if __name__ == '__main__':

    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('experiment_configuration.yml', 'r') as f:
        config = yaml.safe_load(f)
    f.close()
    config['train']['device'] = device
    config['train']['N_EPOCHS'] = 200

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

    trainset = CIFAR10(root="./data", split='train', transform=transform_train, train_ratio=1, download=True)
    testset  = CIFAR10(root="./data", split='test',  transform=transform_test, download=True)
    trainloader = DataLoader(trainset, batch_size=int(config['train']['BATCH_SIZE']), shuffle=True, pin_memory=True, num_workers=1)
    testloader  = DataLoader(testset, batch_size=int(config['train']['BATCH_SIZE']))

    # For resnet18
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=10).to(device)
    model = ResNet18(num_classes=10).to(device)

    model_trainer = TRAINER(model=model, config=config)
    model_trainer.train(trainloader, testloader)

    result_dict = model_trainer.eval(testloader)
    result_dict['model_state_dict'] = model_trainer.model.state_dict()
    result_dict['config'] = config

    time_stamp = datetime.today().strftime("%Y%m%d%H%M%S")
    result_file = f"../clean_models/cifar10_resnet18_clean_{time_stamp}.pkl"
    with open(result_file, 'wb') as f:
        pkl.dump(result_dict, f)
    f.close()
