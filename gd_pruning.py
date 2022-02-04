"""
 * Copyright (C) 2019 Zhonghui You
 * If you are using this code in your research, please cite the paper:
 * Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks, in NeurIPS 2019.

*** TransTailer Algorithm ***

  Input : 
    - Pre-trained Model (pretrained_model), 
    - Target Data (target_data)
  Output: Optimal Sub-Model

"""
import timm
import torch
import os
import numpy as np
import torch.optim as optim
import uuid
import torch.nn as nn
import random
import argparse
from utils import Logger

#region sung util
from data.dataset import load_data
from utility.utils import config

#endregion

#region Gate Decorator
from gate_decorator import *

def set_seeds():
  torch.manual_seed(0)
  torch.cuda.manual_seed_all(0)
  torch.backends.cudnn.deterministic = True
  np.random.seed(0)
  random.seed(0)


def get_model(class_num):
  model = timm.create_model('resnet50')
  model.fc = torch.nn.Linear(2048, class_num)
  
  model.cuda()
  model = torch.nn.DataParallel(model)
  
  return model

def clone_model(net):
  model = get_model(net.module.fc.bias.size()[0])
  gbns = GatedBatchNorm2d.transform(model.module)
  model.load_state_dict(net.state_dict())
  return model, gbns

def eval_prune(pack):
  cloned, _ = clone_model(pack.net)
  _ = Conv2dObserver.transform(cloned.module)
  cloned.module.fc = FinalLinearObserver(cloned.module.fc)
  cloned_pack = dotdict(pack.copy())
  cloned_pack.net = cloned
  Meltable.observe(cloned_pack, 0.001)
  Meltable.melt_all(cloned_pack.net)
  flops, params = analyse_model(cloned_pack.net.module, torch.randn(1, 3, 224, 224).cuda())
  del cloned
  del cloned_pack
  
  return flops, params

#endregion

parser = argparse.ArgumentParser()

parser.add_argument('--data', default='cub', help='cub, stanford_cars')
parser.add_argument('--gpu', default="0", help='gpu id')
parser.add_argument('--neptune', action="store_true")

args = parser.parse_args()

cfg = {
  "data": args.data,
  "project": 'raeyo/decremental',
  "token": 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0OTA1Mzk4OS04MWI4LTQ5YjctYTViZi1iZDEyNjFlOWJmMzAifQ=='
}
logger = Logger(cfg, is_neptune=args.neptune)

EXP_DIR = "/data/sung/checkpoint/supernet/base/0"
TARGET_DATA_LIST = [args.data]

option = config(EXP_DIR)
option.get_config_data()
option.get_config_network()
option.get_config_train()
option.get_config_meta()
option.get_config_tune()

set_seeds()

'''GPU Setting'''
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



'''1. Load Model and Data'''
# target data
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
target_train_data = load_data(option, data_type='train')
target_val_data = load_data(option, data_type='val')
train_loader = torch.utils.data.DataLoader(target_train_data, batch_size=128, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(target_val_data, batch_size=128, num_workers=4)


# pretrained model with target fc
pretrained_model = get_model(len(target_idx_list))
state_dict = torch.load(os.path.join(EXP_DIR, 'best_model.pt'))[0]

for key, value in state_dict.items():
  if "fc" in key:
    state_dict[key] = value[target_idx_list]

pretrained_model.module.load_state_dict(state_dict)



'''2. Convert to Meltable(GateDecorator) Model'''
# Initialize GateDecorator Pack
pack = dotdict({
  'net': pretrained_model,
  'train_loader': train_loader,
  'test_loader': val_loader,
  'trainer': NormalTrainer(use_cuda=torch.cuda.is_available()),
  'criterion': nn.CrossEntropyLoss(),
  'optimizer': None,
  'lr_scheduler': None
})


# Gated Batch Norm2d
GBNs = GatedBatchNorm2d.transform(pack.net)
for gbn in GBNs:
  gbn.extract_from_bn()

pack.optimizer = optim.SGD(
  pack.net.parameters() ,
  lr=2e-3,
  momentum=0.9,
  weight_decay=5e-4,
  nesterov=False
)


# Bottleneck set group for resnet50
layers = [
  pack.net.module.layer1,
  pack.net.module.layer2,
  pack.net.module.layer3,
  pack.net.module.layer4
]

for m in layers:
  masks = []
  for mm in m.modules():
    if mm.__class__.__name__ == 'Bottleneck':
      if mm.downsample is not None:
        masks.append(mm.downsample._modules['1'])
      masks.append(mm.bn3)

  group_id = uuid.uuid1()
  
  for mk in masks:
    mk.set_groupid(group_id)

# Base Metric (Flops, Param, Acc)
cloned, _ = clone_model(pack.net)
BASE_FLOPS, BASE_PARAM = analyse_model(cloned.module, torch.randn(1, 3, 224, 224).cuda())
print('%.3f MFLOPS' % (BASE_FLOPS / 1e6))
print('%.3f M' % (BASE_PARAM / 1e6))
del cloned

result = pack.trainer.test(pack)
previous_acc = result['acc@1']
print(result)


'''4. Gate Decorator Pruning'''

# Set Tick trainset: All trainset(cub, car), Subset(Imagenet)
pack.tick_trainset = pack.train_loader

# Set Pruning Agent
prune_agent = IterRecoverFramework(pack=pack, 
                                   masks=GBNs, 
                                   sparse_lambda=0.001, 
                                   flops_eta=0, 
                                   minium_filter = 3)

# Set save point
flops_save_points = set([40, 38, 35, 32, 30])

iter_idx = 0
best_acc = 0.0
while True:

  # Tock(Fine-tune)
  if iter_idx % 10 == 0:
    print('Tocking:')
    prune_agent.tock(lr_min=0.001, lr_max=0.01, tock_epoch=10)

  # Tick(Pruning)
  left_filter = prune_agent.total_filters - prune_agent.pruned_filters
  num_to_prune = int(left_filter * 0.002)
  info = prune_agent.prune(num_to_prune, tick=True, lr=0.001)
  
  
  # Compute Metric
  flops, params = eval_prune(pack)
  info.update({
    'flops': '[%.2f%%] %.3f MFLOPS' % (flops/BASE_FLOPS * 100, flops / 1e6),
    'param': '[%.2f%%] %.3f M' % (params/BASE_PARAM * 100, params / 1e6)
  })
  print('Iter: %d,\t FLOPS: %s,\t Param: %s,\t Left: %d,\t Pruned Ratio: %.2f %%,\t Train Loss: %.4f,\t Test Acc: %.2f' % 
        (iter_idx, info['flops'], info['param'], info['left'], info['total_pruned_ratio'] * 100, info['train_loss'], info['after_prune_test_acc']))


  # Check Metric
  test_acc = info['after_prune_test_acc']
  flops_ratio = flops/BASE_FLOPS * 100
  params_ratio = params/BASE_PARAM * 100

  logger.logging("Acc", test_acc)
  logger.logging("Flops", flops_ratio)
  logger.logging("Params", params_ratio)

  acc_drop = previous_acc - test_acc
  previous_acc = test_acc

  # Save CheckPoint
  for point in [i for i in list(flops_save_points)]:
    if flops_ratio <= point:
      torch.save(pack.net.module.state_dict(), './gd_result/resnet50_{}_ticktock/flops{:.4f}_acc{:.4f}.ckp'.format(args.data, flops_ratio, test_acc))
      flops_save_points.remove(point)

  if test_acc > best_acc:
    torch.save(pack.net.module.state_dict(), './gd_result/resnet50_{}_ticktock/best.ckp'.format(args.data, flops_ratio, test_acc))


  # Check EndSign
  if len(flops_save_points) == 0:
    break
  if acc_drop > 0.3:
    break

  iter_idx += 1
  