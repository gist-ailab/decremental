"""Pipeline of Pytorch Model Train"""

import os
import argparse
import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

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
torch.cuda.set_device(args.gpu)
cudnn.benchmark = True
torch.manual_seed(args.seed)
cudnn.enabled=True
torch.cuda.manual_seed(args.seed)

'''GPU Setting'''
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu


'''Load Data'''
#Load Dataset
train_dataset, val_dataset = load_dataset(cfg)

#Loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=1)

'''Load Model'''
model = None

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
#TODO: metric
best_epoch, best_score = 0, 0


for epoch in tqdm(range(cfg["maximum_epoch"])):
  model.train()
  
  for batch_idx, (inputs, targets) in enumerate(train_loader):
    optimizer.zero_grad()

    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = 0.
    loss += criterion(outputs, targets)
    
    loss.backward()
    optimizer.step()

  # evaluate
  if epoch % cfg["test_interval"] == 0:
    model = model.eval()
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = 0.
        loss += criterion(outputs, targets)

