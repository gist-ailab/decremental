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

parser.add_argument('--config', default='resnet50_cifar100', help='configuration name')
parser.add_argument('--gpu', default="0", help='gpu id')

args = parser.parse_args()

'''Load Configuration'''
print("EVALUATE {} EXPERIMENT!".format(args.config.upper()))
cfg = load_config('configs/{}.yaml'.format(args.config))
log_dir = os.path.join(cfg["log_dir"], args.config)

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
data_cfg = load_config('configs/resnet50_cifar20_fix.yaml')
#Load Dataset
val_dataset = load_dataset(data_cfg, return_train=False)

#Loader
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=4)

'''Load Model'''
# pretrained model
cfg["num_class"] = 20
model = load_model(cfg)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# load state
state_dict = torch.load(os.path.join(log_dir, "best.pkl"))

for key, value in state_dict.items():
  # for evaluate fc-100
  # if "fc" in key:
  #   temp = torch.zeros_like(value)
  #   temp[:len(data_cfg["selected_class"])] = value[data_cfg["selected_class"]]
  #   not_selected = list(set(range(100)) - set(data_cfg["selected_class"]))
  #   temp[len(data_cfg["selected_class"]):] = value[not_selected]
  #   state_dict[key] = temp
  # for evaluate fc-20
  if "fc" in key:
    state_dict[key] = value[data_cfg["selected_class"]]


model.load_state_dict(state_dict)

'''Criterion'''
criterion = torch.nn.CrossEntropyLoss()

'''evaluate'''
val_acc, val_loss = eval_loop(model=model,
                              data_loader=val_loader,
                              loss_function=criterion)

print('VALID [loss - {:.4f}, acc - {:.4f}]'.format(val_loss, val_acc))