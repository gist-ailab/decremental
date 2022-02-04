from torchvision.transforms import transforms
import torchvision
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from random import shuffle
import torch
import random
from data.custom_dataset import CUB, Stanford_Cars

def load_cifar10(option):
    tr_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    tr_dataset = torchvision.datasets.CIFAR10(root=os.path.join(option.result['data']['data_dir'], 'cifar10'),  train=True, download=True, transform=tr_transform)
    val_dataset = torchvision.datasets.CIFAR10(root=os.path.join(option.result['data']['data_dir'], 'cifar10'), train=False, download=True, transform=val_transform)
    return tr_dataset, val_dataset


def load_cifar100(option):
    tr_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    tr_dataset = torchvision.datasets.CIFAR100(root=os.path.join(option.result['data']['data_dir'], 'cifar100'), train=True, download=True, transform=tr_transform)
    val_dataset = torchvision.datasets.CIFAR100(root=os.path.join(option.result['data']['data_dir'], 'cifar100'), train=False, download=True, transform=val_transform)
    return tr_dataset, val_dataset

def load_imagenet(option, data_type):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tr_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    if data_type=='train':
        tr_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'imagenet', 'train'), transform=tr_transform)
        val_dataset = None
    elif data_type=='val':
        tr_dataset = None
        val_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'imagenet', 'val'), transform=val_transform)
    
    return tr_dataset, val_dataset

def load_tiny_imagenet(option):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tr_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        normalize,
    ])

    tr_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'tiny_imagenet', 'train'), transform=tr_transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'tiny_imagenet', 'val'), transform=val_transform)
    return tr_dataset, val_dataset


def load_cub(option, data_type):
    root = os.path.join(option.result['data']['data_dir'], 'cub')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    tr_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    if data_type=='train':
        tr_dataset = CUB(root, train=True, transform=tr_transform)
        val_dataset = None
        
    elif data_type=='val':
        tr_dataset = None
        val_dataset = CUB(root, train=False, transform=val_transform)

    return tr_dataset, val_dataset


def load_stanford_cars(option, data_type):
    root = os.path.join(option.result['data']['data_dir'], 'stanford_cars')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        
    tr_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    if data_type=='train':
        tr_dataset = Stanford_Cars(root, transform=tr_transform, train=True)
        val_dataset = None
        
    elif data_type=='val':
        tr_dataset = None
        val_dataset = Stanford_Cars(root, transform=val_transform, train=False)
    
    return tr_dataset, val_dataset


def load_flowers(option):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tr_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    tr_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'flowers', 'train'), transform=tr_transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'flowers', 'val'), transform=val_transform)
    return tr_dataset, val_dataset


def load_sketches(option):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tr_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    tr_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'sketches', 'train'), transform=tr_transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'sketches', 'val'), transform=val_transform)
    return tr_dataset, val_dataset   
    


def load_food101(option):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tr_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    tr_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'food101', 'train'), transform=tr_transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'food101', 'val'), transform=val_transform)
    return tr_dataset, val_dataset   

def load_uec256(option):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tr_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    tr_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'train'), transform=tr_transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'val'), transform=val_transform)
    return tr_dataset, val_dataset 


def load_food1k(option):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tr_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    tr_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'train'), transform=tr_transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'val'), transform=val_transform)
    return tr_dataset, val_dataset 

def load_eurosat(option):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tr_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    tr_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'train'), transform=tr_transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'val'), transform=val_transform)
    return tr_dataset, val_dataset 
    
def load_data(option, data_type='train'):
    tr_data_list, val_data_list = [], []
    
    for tr_data, _ in option.result['data']['data_list']:
        if tr_data == 'imagenet':
            tr_d, val_d = load_imagenet(option, data_type)
        elif tr_data == 'tiny_imagenet':
            tr_d, val_d = load_tiny_imagenet(option)
        elif tr_data == 'cub':
            tr_d, val_d = load_cub(option, data_type)
        elif tr_data == 'flowers':
            tr_d, val_d = load_flowers(option)
        elif tr_data == 'sketches':
            tr_d, val_d = load_sketches(option)
        elif tr_data == 'stanford_cars':
            tr_d, val_d = load_stanford_cars(option, data_type)
        elif tr_data == 'food101':
            tr_d, val_d = load_food101(option)
        elif tr_data == 'uec256':
            tr_d, val_d = load_uec256(option)
        elif tr_data == 'food1k':
            tr_d, val_d = load_food1k(option)
        elif tr_data == 'eurosat':
            tr_d, val_d = load_eurosat(option)
        else:
            raise('select appropriate dataset')

        tr_data_list.append(tr_d)
        val_data_list.append(val_d)

    

    if data_type == 'train':
        return MergeSet(tr_data_list)
    else:
        return MergeSet(val_data_list)


class MergeSet(Dataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        self.label_dict = {}
        
        start = 0
        self.targets = []
        self.dataset_index = []
        for ix, dataset in enumerate(self.dataset_list):
            label_list = np.array(dataset.targets)
            self.targets.append(label_list + start)
            
            self.dataset_index += [(ix, i) for i in range(len(label_list))]
            start += len(np.unique(np.array(dataset.targets)))
            
        self.targets = np.concatenate(self.targets, axis=0)
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, index):
        image, _ = self.dataset_list[self.dataset_index[index][0]].__getitem__(self.dataset_index[index][1])
        label = self.targets[index]
        return image, label