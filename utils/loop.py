import torch
from tqdm import tqdm



def train_loop(model, optimizer, data_loader, loss_function, epoch):
  model.train()
  total_loss = 0
  total = 0
  correct = 0
  for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader)):
    optimizer.zero_grad()

    inputs, targets = inputs.cuda(), targets.cuda()
    outputs = model(inputs)
    loss = 0.
    loss += loss_function(outputs, targets)
    
    loss.backward()
    optimizer.step()

    total_loss += loss
    total += targets.size(0)
    _, predicts = outputs.max(1)
    correct += predicts.eq(targets).sum().item()
    optimizer.update(epoch)

  acc = correct/total
  loss = total_loss/len(data_loader)
  
  return acc, loss

@torch.no_grad()
def eval_loop(model, data_loader, loss_function):
  model = model.eval()
  total = 0
  correct = 0
  total_loss = 0
  for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader)):
    inputs, targets = inputs.cuda(), targets.cuda()
    outputs = model(inputs)
    loss = 0.
    loss += loss_function(outputs, targets)
    
    total_loss += loss
    total += targets.size(0)
    _, predicts = outputs.max(1)
    correct += predicts.eq(targets).sum().item()
  acc = correct/total
  loss = total_loss/len(data_loader)

  return acc, loss

