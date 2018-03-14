from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import config as cf
import torchvision
import time
import copy
import os
import sys

from torchvision import datasets, models, transforms
from networks import *
from torch.autograd import Variable
from tensorboardX import SummaryWriter

use_gpu = torch.cuda.is_available()
writer = SummaryWriter()

data_dir = '/home/yash/hw2_data'

data_transforms = {
    'trainf': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'testf': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['trainf', 'testf']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=1) for x in ['trainf', 'testf']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['trainf', 'testf']}
class_names = image_datasets['trainf'].classes

print("| Loading checkpoint model for test phase...")
assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'

checkpoint = torch.load('checkpoint/'+"alexnet_best"+'.t7')

model = checkpoint['model']

if use_gpu:
    model.cuda()

#Testing the trained network
correct = 0
total = 0
i = 1

for data in dataloaders['testf']:
    
    images, labels = data
    if use_gpu:
        images, labels = images.cuda(), labels.cuda()
    outputs = model(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    writer.add_scalar('data/Test_Accuracy', (correct/total) * 100, i)
    i = i + 1

print('Accuracy of the network on the 800 test images: %d %%' % (100 * correct / total))