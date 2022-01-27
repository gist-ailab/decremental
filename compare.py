import os
import argparse
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import numpy as np

from utils import *
from models import load_model
from PIL import Image

import torchvision.transforms as transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class UnNormalize(object):
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, tensor):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized image.
    """
    for t, m, s in zip(tensor, self.mean, self.std):
      t.mul_(s).add_(m)
      # The normalize code -> t.sub_(m).div_(s)
    return tensor

'''Argument'''
parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default="4", help='gpu id')

args = parser.parse_args()

'''Load Configuration'''
config_list = [
  "resnet50_imagenet100_fix",
  "resnet50_imagenet60_fix",
  "resnet50_imagenet40_fix",
  "resnet50_imagenet20_fix" # first vs last
]

cfg_list = [load_config('configs/{}.yaml'.format(config)) for config in config_list]
log_dir_list = [os.path.join(cfg["log_dir"], config) for cfg, config in zip(cfg_list, config_list)]

# cifar
# mean=[0.49139968, 0.48215827, 0.44653124]
# std=[0.24703233, 0.24348505, 0.26158768]

# imagenet
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

inv_normalize = UnNormalize(torch.Tensor(mean), torch.Tensor(std))

'''Seed'''
np.random.seed(cfg_list[-1]["seed"])
cudnn.benchmark = True
torch.manual_seed(cfg_list[-1]["seed"])
cudnn.enabled=True
torch.cuda.manual_seed(cfg_list[-1]["seed"])

'''GPU Setting'''
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu


'''Load Data'''
#Load Dataset
val_dataset = load_dataset(cfg_list[-1], return_train=False)

#Loader
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=4)

'''Load Model'''
# pretrained model
model_list = [load_model(cfg).eval() for cfg in cfg_list]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# target_layers = [model1.conv5_x[-1]]
cam_list = []
for model in model_list:
  model.to(device)
  target_layers = [model.layer4[-1]]
  cam_list.append(GradCAM(model=model, target_layers=target_layers, use_cuda=True))

OX = {val_dataset.classes[idx]: [] for idx in range(cfg_list[-1]["num_class"])}

# load state
for idx, cfg in enumerate(cfg_list):
  if cfg["pretrained"] is not None:
    pass
  else:
    state_dict = torch.load(os.path.join(log_dir_list[idx], "best.pkl"))
    model_list[idx].load_state_dict(state_dict)
  
'''Criterion'''
criterion = torch.nn.CrossEntropyLoss()

'''evaluate'''
total = 0
correct = 0
for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
  inputs, targets = inputs.cuda(), targets.cuda()
  
  outputs1 = model_list[0](inputs)
  outputs2 = model_list[-1](inputs)
  total += targets.size(0)
  
  _, predicts1 = outputs1.max(1)
  _, predicts2 = outputs2.max(1)
  
  correct1_mask = predicts1.eq(targets)
  correct2_mask = predicts2.eq(targets)
  
  OX_mask = torch.logical_and(correct1_mask, ~correct2_mask)
  for input_img, target in zip(inputs[OX_mask], targets[OX_mask]):
    targets = [ClassifierOutputTarget(target.item())]
    OX[val_dataset.classes[target]].append(
      (inv_normalize(input_img.cpu()), [
       cam(input_tensor=input_img.reshape(1, 3, 224, 224), targets=targets) for cam in cam_list
      ])
      )
for class_name, imgs in OX.items():
  print(class_name, len(imgs))
  class_imgs = []
  for input_img, gray_list in imgs:
    vis_list = []
    input_rgb = np.array(input_img.permute(1,2,0))
    
    for gray in gray_list:
      vis_list.append(show_cam_on_image(input_rgb, gray[0, :], use_rgb=True))
    vis_list.append(np.uint8(input_rgb*255))
    class_imgs.append(np.hstack(vis_list))
  if len(class_imgs)>0:
    im = Image.fromarray(np.vstack(class_imgs))
    im.save("{}.png".format(class_name))



