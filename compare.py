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

parser.add_argument('--gpu', default="2", help='gpu id')

args = parser.parse_args()

'''Load Configuration'''
config1 = "resnet50_cifar100"
config2 = "resnet50_cifar20_fix"

print("Compare {} vs {} !".format(config1.upper(), config2.upper()))

cfg1 = load_config('configs/{}.yaml'.format(config1))
cfg2 = load_config('configs/{}.yaml'.format(config2))

log_dir1 = os.path.join(cfg1["log_dir"], config1)
log_dir2 = os.path.join(cfg2["log_dir"], config2)



mean=[0.49139968, 0.48215827, 0.44653124]
std=[0.24703233, 0.24348505, 0.26158768]

inv_normalize = UnNormalize(torch.Tensor(mean), torch.Tensor(std))


'''Seed'''
cfg1["seed"] = cfg2["seed"]

np.random.seed(cfg2["seed"])
cudnn.benchmark = True
torch.manual_seed(cfg2["seed"])
cudnn.enabled=True
torch.cuda.manual_seed(cfg2["seed"])

'''GPU Setting'''
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu


'''Load Data'''
#Load Dataset
val_dataset = load_dataset(cfg2, return_train=False)

#Loader
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=4)

'''Load Model'''
# pretrained model
model1 = load_model(cfg1)
model2 = load_model(cfg2)

model1.eval()
model2.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = model1.to(device)
model2 = model2.to(device)

target_layers = [model1.conv5_x[-1]]
cam1 = GradCAM(model=model1, target_layers=target_layers, use_cuda=True)
target_layers = [model2.conv5_x[-1]]
cam2 = GradCAM(model=model2, target_layers=target_layers, use_cuda=True)

OX = {val_dataset.classes[idx]: [] for idx in range(cfg2["num_class"])}


# load state
if cfg1["pretrained"] is not None:
  pass
else:
  state_dict = torch.load(os.path.join(log_dir1, "best.pkl"))
  model1.load_state_dict(state_dict)
if cfg2["pretrained"] is not None:
  pass
else:
  state_dict = torch.load(os.path.join(log_dir2, "best.pkl"))
  model2.load_state_dict(state_dict)
  
'''Criterion'''
criterion = torch.nn.CrossEntropyLoss()

'''evaluate'''
total = 0
correct = 0
for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
  inputs, targets = inputs.cuda(), targets.cuda()
  outputs1 = model1(inputs)
  outputs2 = model2(inputs)
  
  total += targets.size(0)
  _, predicts1 = outputs1.max(1)
  _, predicts2 = outputs2.max(1)
  
  
  correct1_mask = predicts1.eq(targets)
  correct2_mask = predicts2.eq(targets)
  
  OX_mask = torch.logical_and(correct1_mask, ~correct2_mask)
  for input_img, target in zip(inputs[OX_mask], targets[OX_mask]):
    targets = [ClassifierOutputTarget(target.item())]
    OX[val_dataset.classes[target]].append(
      (inv_normalize(input_img.cpu()), 
       cam1(input_tensor=input_img.reshape(1, 3, 32, 32), targets=targets),
       cam2(input_tensor=input_img.reshape(1, 3, 32, 32), targets=targets))
      )
    
for class_name, imgs in OX.items():
  print(class_name, len(imgs))
  class_imgs = []
  for input_img, gray1, gray2 in imgs:
    input_rgb = np.array(input_img.permute(1,2,0))
    vis1 = show_cam_on_image(input_rgb, gray1[0, :], use_rgb=True)
    vis2 = show_cam_on_image(input_rgb, gray2[0, :], use_rgb=True)
    class_imgs.append(np.hstack([np.uint8(input_rgb*255), vis1, vis2]))
  im = Image.fromarray(np.vstack(class_imgs))
  im.save("{}.png".format(class_name))



