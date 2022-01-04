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
parser.add_argument('--resume', action='store_true', help='resume')

args = parser.parse_args()


'''Load Configuration'''
config_path = 'configs/{}.yaml'.format(args.config)
cfg = load_config(config_path)
print(cfg)

'''Logging'''
logger = Logger(cfg, is_neptune=args.neptune)

log_dir = os.path.join(cfg["log_dir"], args.config)
if not os.path.isdir(log_dir):
  os.mkdir(log_dir)

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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=4)

'''Load Model'''
model = resnet50()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
  print("Using {} gpu!".format(torch.cuda.device_count()))
  model = nn.DataParallel(model)
model = model.to(device)

# load state
if args.resume:
  state_dict = torch.load(os.path.join(log_dir, "cifar100.pkl"))
  model.load_state_dict(state_dict)

'''Optimizer'''
#Adam
# optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
#SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

#TODO: Scheduler
train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["milestone"], gamma=0.2) #learning rate decay
iter_per_epoch = len(train_loader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)


'''Criterion'''
criterion = torch.nn.CrossEntropyLoss()


'''Train'''
best_acc = 0
for epoch in range(cfg["maximum_epoch"]):
  if epoch > 0:
    train_scheduler.step(epoch)

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
    print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d), LR: %.6f'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total, optimizer.param_groups[0]['lr']), end = '')
    if epoch <= 0:
      warmup_scheduler.step()
  train_accuracy = correct/total
  train_avg_loss = total_loss/len(train_loader)

  logger.logging("train/acc", train_accuracy)
  logger.logging("train/loss", train_avg_loss)
  logger.logging("lr", optimizer.param_groups[0]['lr'])

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
    logger.logging("val/acc", val_acc)
    logger.logging("val/loss", val_loss)
    if best_acc < val_acc:
      torch.save(model.state_dict(), os.path.join(log_dir, "cifar100.pkl"))
  print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]'.format(epoch, train_avg_loss, train_accuracy, val_acc))