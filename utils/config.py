import os
import yaml
import torchvision.transforms as transforms
import random


NEPTUNE_PROJECT = 'raeyo/decremental'
NEPTUNE_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0OTA1Mzk4OS04MWI4LTQ5YjctYTViZi1iZDEyNjFlOWJmMzAifQ=='


def load_config(config_path):
  assert os.path.isfile(config_path), "No Configure file in {}".format(config_path)
  with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)
  
  #seed
  random.seed(cfg["seed"])

  # neptune
  cfg["project"] = NEPTUNE_PROJECT
  cfg["token"] = NEPTUNE_TOKEN

  # add transform
  cfg["transform"] = get_transform()

  # decremental classes
  cfg["selected_class"] = get_target_class()(cfg)

  return cfg  

# dataset config
class get_transform:
  data_normalize = {
    "cifar": transforms.Normalize(
      mean=[0.49139968, 0.48215827, 0.44653124], 
      std=[0.24703233, 0.24348505, 0.26158768]),
    "imagenet": transforms.Normalize(
      mean=[0.485, 0.456, 0.406], 
      std=[0.229, 0.224, 0.225])
  }

  @classmethod
  def __call__(cls, dataset='cifar'):
    if dataset=='cifar':
      train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        cls.data_normalize[dataset]])
      valid_transform = transforms.Compose([
        transforms.ToTensor(),
        cls.data_normalize[dataset]])
      return train_transform, valid_transform
    elif dataset=='imagenet':
      train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
          brightness=0.4,
          contrast=0.4,
          saturation=0.4,
          hue=0.2),
        transforms.ToTensor(),
        cls.data_normalize[dataset]])
      valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        cls.data_normalize[dataset]])
      return train_transform, valid_transform
    else:
      raise NotImplementedError

class get_target_class:
  @classmethod
  def __call__(cls, cfg):
    if not 'decre' in cfg["dataset"]:
      original_data = cfg["dataset"]
    original_data = cfg["dataset"].split("_")[-1]
    # original dataset class num
    if original_data=='cifar10':
      original_num = 10
    elif original_data=='cifar100':
      original_num = 100
    elif original_data=='imagenet':
      original_num = 1000
    else:
      raise NotImplementedError
    assert original_num >= cfg["num_class"], "Original Class smaller than target class"

    if cfg["random_order"]:
      selected_class = random.sample(range(original_num), cfg["num_class"])
    else:
      selected_class = list(range(cfg["num_class"]))
  
    return selected_class