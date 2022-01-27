import os
import torch
from torch._C import _debug_set_fusion_group_inlining
import torch.nn as nn
from torchvision.models import resnet as resnet_imagenet

from utils import resnet_cifar

def resnet50_cifar100():
  return resnet_cifar.resnet50()
def resnet50_cifar80():
  return resnet_cifar.resnet50(num_classes=80)
def resnet50_cifar60():
  return resnet_cifar.resnet50(num_classes=60)
def resnet50_cifar40():
  return resnet_cifar.resnet50(num_classes=40)
def resnet50_cifar20():
  return resnet_cifar.resnet50(num_classes=20)
def resnet50_cifar10():
  return resnet_cifar.resnet50(num_classes=10)

def resnet50_imagenet():
  return resnet_imagenet.resnet50()
def resnet50_imagenet20():
  return resnet_imagenet.resnet50(num_classes=20)
def resnet50_imagenet40():
  return resnet_imagenet.resnet50(num_classes=40)
def resnet50_imagenet60():
  return resnet_imagenet.resnet50(num_classes=60)
def resnet50_imagenet80():
  return resnet_imagenet.resnet50(num_classes=80)

def resnet50_imagenet100():
  return resnet_imagenet.resnet50(num_classes=100)
def resnet50_imagenet200():
  return resnet_imagenet.resnet50(num_classes=200)
def resnet50_imagenet400():
  return resnet_imagenet.resnet50(num_classes=400)
def resnet50_imagenet600():
  return resnet_imagenet.resnet50(num_classes=600)
def resnet50_imagenet800():
  return resnet_imagenet.resnet50(num_classes=800)

available_model = {
  "resnet50_cifar100": resnet50_cifar100,
  "resnet50_cifar80": resnet50_cifar80,
  "resnet50_cifar60": resnet50_cifar60,
  "resnet50_cifar40": resnet50_cifar40,
  "resnet50_cifar20": resnet50_cifar20,
  "resnet50_cifar10": resnet50_cifar10,
  
  "resnet50_imagenet": resnet50_imagenet,
  "resnet50_imagenet20": resnet50_imagenet20,
  "resnet50_imagenet40": resnet50_imagenet40,
  "resnet50_imagenet60": resnet50_imagenet60,
  "resnet50_imagenet80": resnet50_imagenet80,
  "resnet50_imagenet100": resnet50_imagenet100,
  "resnet50_imagenet200": resnet50_imagenet200,
  "resnet50_imagenet400": resnet50_imagenet400,
  "resnet50_imagenet600": resnet50_imagenet600,
  "resnet50_imagenet800": resnet50_imagenet800,
}

def load_model(cfg):
  if cfg["pretrained"] is not None:
    model = available_model[cfg["model"]]()
    state_dict = torch.load(os.path.join(cfg["log_dir"], cfg["pretrained"], "best.pkl"))
    for key, value in state_dict.items():
      if "fc" in key:
        state_dict[key] = value[cfg["selected_class"]]
    model.load_state_dict(state_dict)
    return model
  else:
    return available_model[cfg["model"]]()
  