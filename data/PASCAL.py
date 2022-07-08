
import os
import random
from cv2 import NORM_TYPE_MASK

import torch
import torchvision
import cv2
import numpy as np
import scipy.stats as st
from skimage.metrics import structural_similarity
import xml.etree.ElementTree as ET


class PASCAL(torch.utils.data.Dataset):
    
    sets = [('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # each class will be uniquely used as trigger for one target class
    ct_map = {k:int(v) for v, k in enumerate(classes)} 
    
    def __init__(self, 
                 root: str, 
                 transform: torchvision.transforms = None, 
                 target_transform: torchvision.transforms = None) -> None:
        
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        self.data = []
        self.labels = []
        for year, image_set in self.sets:
            
            image_ids = open(os.path.join(self.root, f'VOCdevkit/VOC{year}/ImageSets/Main/{image_set}.txt')).read().strip().split()
            
            for image_id in image_ids:
                
                in_file = open(os.path.join(self.root, f'VOCdevkit/VOC{year}/Annotations/{image_id}.xml'))
                tree = ET.parse(in_file)
                root = tree.getroot()

                self.data.append(os.path.join(self.root, f'VOCdevkit/VOC{year}/JPEGImages/{image_id}.jpg'))
                self.labels.append(self.ct_map[root.find('object').find('name').text])
                
                self.active_data = self.data
                self.active_labels = self.labels
    
    def __len__(self):
        return len(self.active_data)
    
    def __getitem__(self, ind):
        
        img = cv2.imread(self.active_data[ind], cv2.IMREAD_COLOR)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        labels = self.active_labels[ind]
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            labels = self.target_transform(self.active_labels[ind])
            
        return ind, img, labels
    
    
    def select_data(self, ind) -> None : 
        self.active_data = [self.data[i] for i in ind]
        self.active_labels = [self.labels[i] for i in ind]
    
    
if __name__ == '__main__':
    
    test = PASCAL(root = '/scr/songzhu/pascal')
    
    for _, (ind, img, labels) in enumerate(test):
        
        print(img)