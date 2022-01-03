"""Pipeline of Pytorch Model Train"""

import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import timm

import numpy as np

from utils import *

'''Argument'''
parser = argparse.ArgumentParser()

parser.add_argument('--config', default='default', help='configuration name')
parser.add_argument('--gpu', default="0", help='gpu id')
parser.add_argument('--neptune', action='store_true', help='logging to neptune')

args = parser.parse_args()


'''Load Configuration'''
config_path = 'configs/{}.yaml'.format(args.config)
cfg = load_config(config_path)
print(cfg)

'''Logging'''
logger = Logger(cfg, neptune=args.neptune)

'''Seed'''
np.random.seed(cfg["seed"])
cudnn.benchmark = True
torch.manual_seed(cfg["seed"])
cudnn.enabled=True
torch.cuda.manual_seed(cfg["seed"])

'''GPU Setting'''
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu


'''Load Data'''
#Load Dataset
train_dataset, val_dataset = load_dataset(cfg)

#Loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, num_workers=1)

'''Load Model'''
model = resnet50()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
  print("Using {} gpu!".format(torch.cuda.device_count()))
  model = nn.DataParallel(model)
model = model.to(device)

'''Optimizer'''
#Adam
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

#TODO: Scheduler

'''Criterion'''
criterion = torch.nn.CrossEntropyLoss()


'''Train'''
best_acc = 0
for epoch in tqdm(range(cfg["maximum_epoch"])):
  model.train()
  total_loss = 0
  total = 0
  correct = 0
  for batch_idx, (inputs, targets) in enumerate(train_loader):
    optimizer.zero_grad()

    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = 0.
    loss += criterion(outputs, targets)
    
    loss.backward()
    optimizer.step()

    total_loss += loss
    total += targets.size(0)
    _, predicts = outputs.max(1)
    correct += predicts.eq(targets).sum().item()
    print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')
  train_accuracy = correct/total
  train_avg_loss = total_loss/len(train_loader)

  logger.logging("train/acc", train_accuracy)
  logger.logging("train/loss", train_avg_loss)

  # evaluate
  if epoch % cfg["test_interval"] == 0:
    model = model.eval()
    
    total = 0
    correct = 0
    total_loss = 0
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = 0.
        loss += criterion(outputs, targets)
        
        total_loss += loss
        total += targets.size(0)
        _, predicts = outputs.max(1)
        correct += predicts.eq(targets).sum().item()
    val_acc = correct/total
    val_loss = total_loss/len(val_loader)
    logger.logging("val/acc", train_accuracy)
    logger.logging("val/loss", train_avg_loss)
    if best_acc < val_acc:
      torch.save(model.state_dict(), "cifar100.pkl")
  print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]'.format(epoch, train_avg_loss, train_accuracy, val_acc))