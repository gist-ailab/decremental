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

parser.add_argument('--config', default='resnet50_yyg', help='configuration name')
parser.add_argument('--gpu', default="4", help='gpu id')

args = parser.parse_args()

'''Load Configuration'''
config_path = 'configs/{}.yaml'.format(args.config)
cfg = load_config(config_path)
print(cfg)
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
#Load Dataset
_, val_dataset, selected_class = load_dataset(cfg, return_class=True)

#Loader
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=4)

'''Load Model'''
# pretrained model
model = resnet50(num_classes=cfg["num_class"]) 
model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
  print("Using {} gpu!".format(torch.cuda.device_count()))
  model = nn.DataParallel(model)
model = model.to(device)


# load state
state_dict = torch.load(os.path.join(log_dir, "best.pkl"))
for key, value in state_dict.items():
  if "fc" in key:
    # state_dict[key] = value[:cfg["num_class"]] # 
    # state_dict[key] = value[:100] # 
    state_dict[key] = value[selected_class] # 
    
model.load_state_dict(state_dict)


'''Criterion'''
criterion = torch.nn.CrossEntropyLoss()

'''evaluate'''
val_acc, val_loss = eval_loop(model=model,
                              data_loader=val_loader,
                              loss_function=criterion)

print('VALID [loss - {:.4f}, acc - {:.4f}]'.format(val_loss, val_acc))