import torchvision.transforms as transforms
import torchvision.datasets as dset

import os
import random
import numpy as np


#region public dataset

def cifar10(cfg, return_train=True):
  root = cfg["root"]
  train_transform, valid_transform = cfg["transform"]('cifar')
  valid_dataset = dset.CIFAR10(root=root, train=False, download=True, transform=valid_transform)
  if return_train:
    train_dataset = dset.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    return train_dataset, valid_dataset
  else:
    return valid_dataset

def cifar100(cfg, return_train=True):
  root = cfg["root"]
  train_transform, valid_transform = cfg["transform"]('cifar')
  valid_dataset = dset.CIFAR100(root=root, train=False, download=True, transform=valid_transform)
  if return_train:
    train_dataset = dset.CIFAR100(root=root, train=True, download=True, transform=train_transform)
    return train_dataset, valid_dataset
  else:
    return valid_dataset

def imagenet(cfg, return_train=True):
  root = cfg["root"]
  train_transform, valid_transform = cfg["transform"]('imagenet')
  valid_dataset = dset.ImageFolder(
    root=os.path.join(root, "val"),
    train=False, download=True, transform=valid_transform)
  if return_train:
    train_dataset = dset.ImageFolder(
      root=os.path.join(root, "train"),
      train=True, download=True, transform=train_transform)
    return train_dataset, valid_dataset
  else:
    return valid_dataset


#endregion

#region custom(decremental) dataset
class DecreCifar100(dset.CIFAR100):
  """
  reduce class 100 ==> N
  - by selected class
  """
  def __init__(self, cfg, **kwargs):
    super(DecreCifar100, self).__init__(**kwargs)

    self.selected_class = cfg["selected_class"]
    self.selected_class.sort()
    self.reduce_class()
    print("Target Class is >>>".format(len(self.selected_class)))
    for class_name, index in self.class_to_idx.items():
      print("CLASS {:<15} | ID: {:<3} | ORIGINAL ID:{:<3}".format(class_name, index,
                                                           self.idx_to_orginal[index]))
        

  def reduce_class(self):
    self.original_to_target = {}

    mask = np.full_like(self.targets, False, dtype=bool)
    orginal_targets = np.array(self.targets)
    for target_class in self.selected_class:
      mask = np.logical_or(mask, orginal_targets == target_class)
    
    reduced_data = self.data[mask]
    unique_labels, reduced_targets = np.unique(orginal_targets[mask], return_inverse=True)
    
    reduced_class_to_idx = {}
    idx_to_orginal = {}
    for name, ind in self.class_to_idx.items():
      if ind in unique_labels:
        reduced_class_to_idx[name] = self.selected_class.index(ind)
        idx_to_orginal[self.selected_class.index(ind)] = ind
    self.idx_to_orginal = idx_to_orginal
    
    self.data = reduced_data
    self.targets = reduced_targets
    self.class_to_idx = reduced_class_to_idx
    self.classes = list(reduced_class_to_idx.keys())

def decre_cifar100(cfg, return_train=True):
  root = cfg["root"]
  train_transform, valid_transform = cfg["transform"]('cifar')
  valid_dataset = DecreCifar100(cfg, root=root, train=False, download=True, transform=valid_transform)
  
  if return_train:
    train_dataset = DecreCifar100(cfg, root=root, train=True, download=True, transform=train_transform)
    return train_dataset, valid_dataset
  else:
    return valid_dataset

#endregion


available_dataset = {
  # public dataset
  "cifar10": cifar10,
  "cifar100": cifar100,
  "imagenet": imagenet,

  # custom dataset
  "decre_cifar100": decre_cifar100,
}

def load_dataset(cfg, return_train=True):
  assert cfg["dataset"] in available_dataset.keys(), "{} is Not available dataset".format(cfg["dataset"])

  return available_dataset[cfg["dataset"]](cfg, return_train)

  