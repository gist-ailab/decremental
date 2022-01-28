"""
 * Copyright (C) 2019 Zhonghui You
 * If you are using this code in your research, please cite the paper:
 * Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks, in NeurIPS 2019.

*** TransTailer Algorithm ***

  Input : Pre-trained Model, Source Data, Target Data
  Output: Optimal Sub-Model

"""
import timm
import torch
import os
import numpy as np

from data.dataset import load_data
from utility.utils import config

'''Parameters'''
EXP_DIR = "/data/sung/checkpoint/supernet/base/0"
TARGET_DATA_LIST = [
  # 'imagenet', 
  'cub', 
  'stanford_cars'
  ]

option = config(EXP_DIR)
option.get_config_data()
option.get_config_network()
option.get_config_train()
option.get_config_meta()
option.get_config_tune()



'''GPU Setting'''
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''1. Load Model'''
# pretrained_model = timm.create_model('resnet50')
# state_dict = torch.load(os.path.join(EXP_DIR, 'best_model.pt'))[0]
# pretrained_model.fc = torch.nn.Linear(2048, 1396)
# pretrained_model.load_state_dict(state_dict)

source_data = load_data(option, data_type='val')

start_idx = 0
target_idx_range = []
target_data_list = []
for tr_data, volume in option.result['data']['data_list']:
  if tr_data in TARGET_DATA_LIST:
    target_idx_range.append([start_idx, start_idx+volume])
    target_data_list.append([tr_data, volume])
  start_idx += volume
for idx, target_data in enumerate(TARGET_DATA_LIST):
  print("{} data index is {} to {}".format(target_data, *target_idx_range[idx]))
                                           
target_idx_list = []
for s, e in target_idx_range:
  target_idx_list += list(range(s, e))

option.result['data']['data_list'] = target_data_list
target_data = load_data(option, data_type='val')
print()


'''2. Initialize scaling factor'''



'''2.1 Convert BatchNorm2d -> GatedBatchNorm2d'''

'''3. Initialize FC layer to target data'''

optimal_model = None


'''4. TransTailer Pruning'''

while True:
  #1. Train scaling factor
  #2. Transfrom scaling factor to global importance
  #3. Pruning based on global importance
  #4. Fine-tune for target data
  #5. Check Accuracy Drop
  
  pass
  