import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'

class ConvolutionalModel(nn.Module):
  def __init__(self, in_channels, conv1_width, conv2_width, fc1_width, class_count):
    super(ConvolutionalModel, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding='same', bias=True)
    self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5, stride=1, padding='same', bias=True)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(7*7*32, fc1_width, bias=True)
    self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)
    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear) and m is not self.fc_logits:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    self.fc_logits.reset_parameters()

  def forward(self, x):
    h = self.conv1(x)
    h = self.maxpool(h)
    h = torch.relu(h)
    
    h = self.conv2(h)
    h = self.maxpool(h)
    h = torch.relu(h)
    
    h = h.view(h.shape[0], -1)
    h = self.fc1(h)
    h = torch.relu(h)
    logits = self.fc_logits(h)
    return torch.softmax(logits, dim=1)
  
def train(model, train_loader, val_loader, config):
    writer = SummaryWriter(filename_suffix='conv')
    model.train()

    max_epochs = config['max_epochs']
    weight_decay = config['weight_decay']
    base_lr = config['base_lr']
    step_size = config['step_size']
    gamma = config['gamma']

    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in range(1, max_epochs + 1):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{max_epochs}")
        for i, (x, y) in loop:
            Y_ = model(x)
            loss = nn.functional.cross_entropy(Y_, y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            loop.set_postfix(loss=loss.item())
            writer.add_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + i)

        scheduler.step()
        model.eval()

        with torch.no_grad():
            for i, (x_val, y_val) in enumerate(val_loader):
                val_Y_ = model(x_val)
                loss_val = nn.functional.cross_entropy(val_Y_, y_val)
                writer.add_scalar("Val/Loss", loss_val.item(), epoch * len(val_loader) + i)
        
        model.train()

        writer.add_images("conv1_filters", model.conv1.weight.view(-1, 1, 5, 5), epoch)
        writer.add_images("conv2_filters", model.conv2.weight.view(-1, 1, 5, 5), epoch)
    writer.close()


if __name__=="__main__":
   np.random.seed(int(time.time() * 1e6) % 2**31)

   train_dataset = MNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
   train_size = int(0.8*len(train_dataset))
   ds_test = MNIST(DATA_DIR, train=False, transform=transforms.ToTensor())
   val_size = len(train_dataset) - train_size

   ds_train, ds_val = random_split(train_dataset, [train_size, val_size])

   x_train, y_train = ds_train.dataset.data, ds_train.dataset.targets
   x_train = x_train.float().div_(255.0)
   x_val, y_val = ds_val.dataset.data, ds_val.dataset.targets
   x_val = x_val.float().div_(255.0)
   x_test, y_test = ds_test.data, ds_test.targets
   x_test = x_test.float().div_(255.0)

   train_data = torch.utils.data.TensorDataset(x_train.view(-1, 1, 28, 28), y_train)
   val_data = torch.utils.data.TensorDataset(x_val.view(-1, 1, 28, 28), y_val)
   train_loader = DataLoader(train_data, batch_size=50, shuffle=True)
   val_loader = DataLoader(val_data, batch_size=50, shuffle=False)

   model = ConvolutionalModel(1, 16, 32, 512, 10)

   config = {
        'max_epochs': 8,
        'base_lr': 1e-1,
        'step_size': 2,
        'gamma': 0.1,
        'weight_decay': 1e-2
    }
   
   train(model, train_loader, val_loader, config)
   model.eval()

   with torch.no_grad():
      X = x_test.view(-1, 1, 28, 28)
      Yoh_ = torch.eye(10)[y_test]
      Y_ = model(X)
      loss = nn.functional.cross_entropy(Y_, Yoh_)
      print(f"Test loss = {loss.item()}")
      acc = (torch.argmax(Y_, dim=1) == y_test).float()
      acc = acc.mean()
      print(f"Test accuracy = {acc.item()}")
   
   torch.save(model.state_dict(), 'out/model.pth')


   
