import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from utils import *
from models import load_model

'''Argument'''
parser = argparse.ArgumentParser()

parser.add_argument('--config', default='resnet50_imagenet40_fix', help='configuration name')
parser.add_argument('--gpu', default="4", help='gpu id')
parser.add_argument('--neptune', action='store_true', help='logging to neptune')

args = parser.parse_args()

'''Load Configuration'''
print("START {} EXPERIMENT!".format(args.config.upper()))
cfg = load_config('configs/{}.yaml'.format(args.config))

'''Logging'''
log_dir = os.path.join(cfg["log_dir"], args.config)
if not os.path.isdir(log_dir):
  os.mkdir(log_dir)
logger = Logger(cfg, is_neptune=args.neptune)

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
train_dataset, val_dataset = load_dataset(cfg, return_train=True)

#Loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=4)

'''Load Model'''
model = load_model(cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

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
    torch.save({
      'epoch': epoch,
      'optimzer_state_dict': optimizer.state_dict(),
      "model_state_dict": model.state_dict()
      }, os.path.join(log_dir, "checkpoint_{}.pkl".format(epoch)))
    
  print("=====EPOCH : {} =====".format(epoch))
  print(logger)
