import os
import argparse
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import numpy as np

from utils import *
from models import load_model

'''Argument'''
parser = argparse.ArgumentParser()

parser.add_argument('--config', default='resnet50_imagenet20_fix', help='configuration name')
parser.add_argument('--gpu', default="2", help='gpu id')

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
#Load Dataset
val_dataset = load_dataset(cfg, return_train=False)

#Loader
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=4)

'''Load Model'''
# pretrained model
model = load_model(cfg)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# load state
if cfg["pretrained"] is not None:
  pass
else:
  state_dict = torch.load(os.path.join(log_dir, "best.pkl"))
  model.load_state_dict(state_dict)

'''Criterion'''
criterion = torch.nn.CrossEntropyLoss()

'''evaluate'''
val_acc, val_loss = eval_loop(model=model,
                              data_loader=val_loader,
                              loss_function=criterion)

print('VALID [loss - {:.4f}, acc - {:.4f}]'.format(val_loss, val_acc))