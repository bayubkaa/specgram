'''Train CIFAR10 with PyTorch.'''
from ast import arg
import torch
import torch.nn as nn

from modules.resnet import ResNet50
from modules.mobilenetv2 import MobileNetV2

import os
import argparse
import json

#from models import *
from utils.utils_progbar import progress_bar
from utils.create_dataset import generate_data_loader
from config.read_config import config

import numpy as np

from torch.utils.tensorboard import SummaryWriter
import numpy as np

from a import get_pruned_resnet50

writer = SummaryWriter('runs/EQ')

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

parser.add_argument('--model', action='store', type=str, help='pilih model')
parser.add_argument('--ratio_pruned', action='store', type=float, help='pilih model')

args = parser.parse_args()

if args.model is None:
    pass
    # assert False, "define the model!"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

config = config()['config']
classes = config['classes']
num_class = len(classes)
img_size = config['resize']

num_epochs = config['num_epochs']
batch_size = config['batch_size']
learning_rate = config['learning_rate']

testloader = generate_data_loader(root_dir="data_testing",
                                        annotation_file="data_annotations_testing.csv", 
                                        img_size=img_size, 
                                        batch_size=batch_size,
                                        split_loader=False)   

classes = ('EQ', 'NO')

# Model
print('==> Building model..')

# if args.model == "resnet50":
#     net = ResNet50(num_classes=num_class)
#     print("model: ResNet50")
# elif args.model == "mobilenetv2":
#     net = MobileNetV2(ch_in=3, n_classes=num_class)
#     print("model: MobileNetV2")
# elif args.model == "prune_resnet50":



criterion = nn.CrossEntropyLoss()
def test(net):
    net = net.to(device)
    print(device)
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            average_val_loss = test_loss/(batch_idx+1)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            val_acc = 100.*correct/total
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (average_val_loss, val_acc, correct, total))



if __name__ == "__main__":
    
    # net = ResNet50(num_classes=num_class)
    # checkpoint = torch.load('checkpoint/resnet50_ckpt.pth')
    # net.load_state_dict(checkpoint["net"])
    # test(net)

    # net = get_pruned_resnet50(norm_ord=2, ratio=0.7)
    # checkpoint = torch.load('checkpoint/prune_resnet50_pruned_70_ckpt.pth')
    # net.load_state_dict(checkpoint["net"])
    # test(net)
    
    # net = get_pruned_resnet50(norm_ord=2, ratio=0.9)
    # checkpoint = torch.load('checkpoint/prune_resnet50_pruned_90_ckpt.pth')
    # net.load_state_dict(checkpoint["net"])
    # test(net)

    # net = get_pruned_resnet50(norm_ord=2, ratio=0.95)
    # checkpoint = torch.load('checkpoint/prune_resnet50_pruned_95_ckpt.pth')
    # net.load_state_dict(checkpoint["net"])
    # test(net)

    # net = MobileNetV2(ch_in=3, n_classes=num_class)
    # checkpoint = torch.load('checkpoint/mobilenetv2_ckpt.pth')
    # net.load_state_dict(checkpoint["net"])
    # test(net)

    # from modules.cred import CRED
    # net = CRED(batch_size=batch_size, size=224)
    # checkpoint = torch.load('checkpoint/cred_ckpt.pth')
    # net.load_state_dict(checkpoint["net"])
    # test(net)

    # from modules.hajar import HAJAR
    # net = HAJAR(batch_size=batch_size, size=224)
    # checkpoint = torch.load('checkpoint/hajar_ckpt.pth')
    # net.load_state_dict(checkpoint["net"])
    # test(net)

    # from modules.kaelynn import KAELYNN
    # net = KAELYNN()
    # checkpoint = torch.load('checkpoint/kaelynn_ckpt.pth')
    # net.load_state_dict(checkpoint["net"])
    # test(net)

    
    net = get_pruned_resnet50(norm_ord=2, ratio=0.9)
    checkpoint = torch.load('checkpoint/prune_resnet50_pruned_90__l2norm_ckpt.pth')
    net.load_state_dict(checkpoint["net"])
    test(net)