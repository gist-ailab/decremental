import torchvision.transforms as transforms
import torchvision.datasets as dset

import os

class cifar:
  normalize = transforms.Normalize(
    mean=[0.49139968, 0.48215827, 0.44653124], 
    std=[0.24703233, 0.24348505, 0.26158768])
  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
  ])
  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
  ])

class imagenet:
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
      brightness=0.4,
      contrast=0.4,
      saturation=0.4,
      hue=0.2),
    transforms.ToTensor(),
    normalize,
  ])
  valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
  ])


class cifar10(cifar):
  @classmethod
  def __call__(cls, cfg):
    root = cfg["root"]
    train_dataset = dset.CIFAR10(root=root, train=True, download=True, transform=cls.train_transform)
    valid_dataset = dset.CIFAR10(root=root, train=False, download=True, transform=cls.valid_transform)
    return train_dataset, valid_dataset

class cifar100(cifar):
  @classmethod
  def __call__(cls, cfg):
    root = cfg["root"]
    train_dataset = dset.CIFAR100(root=root, train=True, download=True, transform=cls.train_transform)
    valid_dataset = dset.CIFAR100(root=root, train=False, download=True, transform=cls.valid_transform)
    return train_dataset, valid_dataset

class imageNet(imagenet):
  @classmethod
  def __call__(cls, cfg):
    root = cfg["root"]
    train_dataset = dset.ImageFolder(
      root=os.path.join(root, "train"),
      train=True, download=True, transform=cls.train_transform)
    valid_dataset = dset.ImageFolder(
      root=os.path.join(root, "val"),
      train=False, download=True, transform=cls.valid_transform)
    return train_dataset, valid_dataset


available_dataset = {
  "cifar10": cifar10(),
  "cifar100": cifar100(),
  "imageNet": imageNet(),
}

def load_dataset(cfg):
  return available_dataset[cfg["dataset"]](cfg)

