import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import _LRScheduler



class OptBase:
  
  def __init__(self, optimizer, scheduler=None, warmup=None):
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.warmup = warmup

  def step(self):
    self.optimizer.step()
  def update(self, *args):
    pass
  def zero_grad(self):
    self.optimizer.zero_grad()

  def state_dict(self):
    state_dict = {
      "optimizer": self.optimizer.state_dict(),
    }
    if self.scheduler is None:
      state_dict["scheduler"] = None
    else:
      state_dict["scheduler"] = self.scheduler.state_dict()
    return state_dict
    
  @property
  def last_learning_rate(self):
    return self.optimizer.param_groups[0]['lr']

class AdamOpt(OptBase):
  def __init__(self, parameters, cfg):
    super(AdamOpt, self).__init__(
      optimizer=Adam(parameters, lr=cfg["lr"], weight_decay=cfg["wd"]))  

class SGDOpt(OptBase):
  def __init__(self, parameters, max_iter, cfg):
    optimizer=SGD(parameters, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["wd"])
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["milestone"], gamma=0.2) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer, max_iter)
    self.warmup_epoch = cfg["warmup_epoch"]
    super(SGDOpt, self).__init__(
      optimizer=optimizer,
      scheduler=train_scheduler,
      warmup=warmup_scheduler
      )  
  
  def update(self, epoch):
    if epoch > self.warmup_epoch:
      self.scheduler.step(epoch)
    else:
      self.warmup.step(epoch)


class WarmUpLR(_LRScheduler): # https://github.com/weiaicunzai/pytorch-cifar100/blob/2149cb57f517c6e5fa7262f958652227225d125b/utils.py#L234
  """warmup_training learning rate scheduler
  Args:
    optimizer: optimzier(e.g. SGD)
    total_iters: totoal_iters of warmup phase
  """
  def __init__(self, optimizer, total_iters, last_epoch=-1):
    self.total_iters = total_iters
    super().__init__(optimizer, last_epoch)

  def get_lr(self):
    """we will use the first m batches, and set the learning
    rate to base_lr * m / total_iters
    """
    return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

