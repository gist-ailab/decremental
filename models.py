import os
import torch
import torch.nn as nn
from torchvision.models import resnet as resnet_imagenet

from utils import resnet_cifar

class Score80(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.target_class = cfg["selected_class"]
        self.other_class = list(set(range(100)) - set(self.target_class))            
        self.score_layer = nn.Linear(80, 20)
    def forward(self, x):
        other_score = x[:, self.other_class]
        output = self.score_layer(other_score)
        return output

class ScoreMIX(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.target_class = cfg["selected_class"]
        self.other_class = list(set(range(100)) - set(self.target_class))            
        self.score_layer = nn.Linear(80, 20)
    def forward(self, x):
        target_score = x[:, self.target_class]
        other_score = x[:, self.other_class]
        other_score = torch.relu(self.score_layer(other_score))
        
        output = torch.softmax(other_score, dim=1)
        
        return target_score*output


class ScoreNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.target_class = cfg["selected_class"]
        self.other_class = list(set(range(100)) - set(self.target_class))            
        self.score_layer = nn.Linear(80, 20)
    def forward(self, x):
        other_score = x[:, self.other_class]
        other_score = torch.softmax(other_score, dim=1)
        output = self.score_layer(other_score)
        return output

def resnet50_cifar100():
  return resnet_cifar.resnet50()

def resnet50_cifar20():
  return resnet_cifar.resnet50(num_class=20)

def resnet50_imagenet():
  return resnet_imagenet.resnet50(num_class=20)

available_model = {
  "resnet50_cifar100": resnet50_cifar100,
  "resnet50_cifar20": resnet50_cifar20,
  
  "resnet50_imagenet": resnet50_imagenet,
  
}

# for key, value in state_dict.items():
#   # for evaluate fc-100
#   # if "fc" in key:
#   #   temp = torch.zeros_like(value)
#   #   temp[:len(data_cfg["selected_class"])] = value[data_cfg["selected_class"]]
#   #   not_selected = list(set(range(100)) - set(data_cfg["selected_class"]))
#   #   temp[len(data_cfg["selected_class"]):] = value[not_selected]
#   #   state_dict[key] = temp
#   # for evaluate fc-20
#   if "fc" in key:
    # state_dict[key] = value[data_cfg["selected_class"]]


def load_model(cfg):
  if cfg["pretrained"] is not None:
    state_dict = torch.load(os.path.join(cfg["log_dir"], cfg["pretrained"], "best.pkl"))
  
  else:
    return available_model[cfg["model"]]()
  