from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tensorboardX import SummaryWriter

from modules import train_model

data_dir = '/home/yash/hw2_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['trainf', 'testf']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=1) for x in ['trainf', 'testf']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['trainf', 'testf']}
class_names = image_datasets['trainf'].classes

writer = SummaryWriter()
use_gpu = torch.cuda.is_available()
model = torchvision.models.alexnet(pretrained=True)

#Don't modify the weights while training.
for param in model.parameters():
    param.requires_grad = False
#Parameters of newly constructed modules have requires_grad=True by default

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8)

if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

#Only parameters of final layer are being optimized.
optimizer_conv = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_trained = train_model(model, criterion, optimizer_conv, exp_lr_scheduler, dataloaders, dataset_sizes, 25)

#Testing the trained network
correct = 0
total = 0
i = 1

for data in dataloaders['testf']:
    
    images, labels = data
    outputs = model_trained(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    writer.add_scalar('data/Test_Accuracy', (correct/total) * 100, i)
    i = i + 1

print('Accuracy of the network on the 800 test images: %d %%' % (100 * correct / total))
