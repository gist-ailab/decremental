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
from torchvision.models import resnet50
model = resnet50(num_classes=100)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
  print("Using {} gpu!".format(torch.cuda.device_count()))
  model = nn.DataParallel(model)
model = model.to(device)

# load state
if args.resume:
  state_dict = torch.load(os.path.join(log_dir, "best.pkl"))
  model.load_state_dict(state_dict)

'''Optimizer'''
if cfg["optimizer"] == "Adam":
  optimizer = AdamOpt(model.parameters(), cfg)
elif cfg["optimizer"] == "SGD":
  optimizer = SGDOpt(model.parameters(), len(train_loader), cfg)
else:
  raise NotImplementedError

'''Criterion'''
criterion = torch.nn.CrossEntropyLoss()

'''Train'''
best_epoch, best_acc = 0, 0
for epoch in range(1, cfg["maximum_epoch"]+1):
  train_accuracy, train_avg_loss = train_loop(model=model, 
                                              optimizer=optimizer,
                                              data_loader=train_loader,
                                              loss_function=criterion, epoch=epoch)
  logger.logging("train/acc", train_accuracy)
  logger.logging("train/loss", train_avg_loss)
  logger.logging("lr", optimizer.last_learning_rate)

  # evaluate
  if epoch % cfg["test_interval"] == 0:
    val_acc, val_loss = eval_loop(model=model,
                                  data_loader=val_loader,
                                  loss_function=criterion)

    logger.logging("val/acc", val_acc)
    logger.logging("val/loss", val_loss)
    if best_acc < val_acc:
      best_epoch = epoch
      best_acc = val_acc
      torch.save(model.state_dict(), os.path.join(log_dir, "best.pkl"))
    logger.logging("best/acc", best_acc)
    
  if epoch % 10 == 0:
    torch.save(model.state_dict(), os.path.join(log_dir, "epoch_{}.pkl".format(epoch)))
  print('EPOCH {:4} |TRAIN [loss - {:.4f}, acc - {:.4f}] |VALID [loss - {:.4f}, acc - {:.4f}] |BEST [epoch - {:4}, acc - {:.4f}]'.format(
        epoch, 
        train_avg_loss, train_accuracy, 
        val_loss, val_acc,
        best_epoch, best_acc))