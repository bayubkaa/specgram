'''Train CIFAR10 with PyTorch.'''
from ast import arg
import torch
import torch.nn as nn

from modules.resnet import ResNet18, ResNet50
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

from utils.pruning_utils import get_new_conv, get_new_bn, get_new_dependant_conv, get_model_size
from utils.pruning_utils import pick_filter_to_prune

writer = SummaryWriter('runs/EQ')

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

parser.add_argument('--model', action='store', type=str, help='pilih model')
parser.add_argument('--ratio_pruned', action='store', type=float, help='pilih model')

args = parser.parse_args()

if args.model is None:
    assert False, "define the model!"

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

trainloader, testloader = generate_data_loader(root_dir="data_training",
                                        annotation_file="data_annotations.csv", 
                                        img_size=img_size, 
                                        batch_size=batch_size)   

classes = ('EQ', 'NO')

# Model
print('==> Building model..')
from a import get_pruned_resnet50

if args.model == "resnet50":
    net = ResNet50(num_classes=num_class)
    print("model: ResNet50")
elif args.model == "resnet18":
    net = ResNet18(num_classes=num_class)
    print("model: ResNet18")
elif args.model == "mobilenetv2":
    net = MobileNetV2(ch_in=3, n_classes=num_class)
    print("model: MobileNetV2")
elif args.model == "prune_resnet50":
    args.model = args.model + "_pruned_" + str(int(args.ratio_pruned*100))
    net = get_pruned_resnet50(norm_ord=2, ratio=args.ratio_pruned)
    print(f"model: ResNet50-pruned with ratio {args.ratio_pruned}")
elif args.model == "cred":
    from modules.cred import CRED
    net = CRED(batch_size=batch_size, size=224)
    print("model: CRED")
elif args.model == "hajar":
    from modules.hajar import HAJAR
    net = HAJAR(batch_size=batch_size, size=224)
    print("model: HAJAR")
elif args.model == "kaelynn":
    from modules.kaelynn import KAELYNN
    net = KAELYNN()
    print("model: KAELYNN")
   

#net = ResNet50(num_classes=num_class)#MobileNetV2(ch_in=3, n_classes=num_class)#ResNet50(num_classes=len(classes))
net = net.to(device)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{args.model}_ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
performance = {}
performance["train_loss"] = []
performance["train_acc"] = []
performance["val_loss"] = []
performance["val_acc"] = []

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        outputs = net(inputs)
        #print(outputs.shape)
        loss = criterion(outputs, targets)
        
        #with torch.autograd.set_detect_anomaly(True):
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        average_train_loss = train_loss/(batch_idx+1)

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_acc = 100.*correct/total

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (average_train_loss, train_acc, correct, total))

    performance["train_loss"].append(average_train_loss)
    performance["train_acc"].append(train_acc)

def test(epoch):
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

    performance["val_loss"].append(average_val_loss)
    performance["val_acc"].append(val_acc)
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/{args.model}_ckpt.pth')
        best_acc = acc

def save_metrics_json():
    with open(f"metrics/{args.model}.json", "w") as outfile:
        json.dump(performance, outfile)


for epoch in range(start_epoch, start_epoch+num_epochs):
    save_metrics_json()
    train(epoch)
    test(epoch)
    scheduler.step()


