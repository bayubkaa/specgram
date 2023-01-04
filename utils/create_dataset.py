from torch.utils import data
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import cv2

def int_to_one_hot(y, batch_size, nb_digits):
    y_onehot = torch.FloatTensor(batch_size, nb_digits)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

class CharData(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        #img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        img = cv2.imread(os.path.join(self.root_dir, img_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        if self.transform is not None:
            img = self.transform(img)

        return (img, y_label)

def train_val_dataset(dataset, val_split=0.10):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, random_state=1)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    lenght = (len(datasets['train']), len(datasets['val']))
    print(f'Splitted: train/val = {lenght}')
    return datasets

def get_transform(img_size):
    transform=transforms.Compose([
        transforms.ToPILImage(),
        SquarePad(),
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform

def generate_data_loader(root_dir, annotation_file, img_size, batch_size, transform=None, val_split=0.10, split_loader=True):
    transform=get_transform(img_size=img_size)
    dataset = CharData(root_dir=root_dir, annotation_file=annotation_file,
                transform=transform)
    if split_loader:
        dataset = train_val_dataset(dataset, val_split=val_split)
        train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size,
                                                shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset["val"], batch_size=batch_size,
                                                shuffle=True)

        #data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        #                                       shuffle=True)
        return train_loader, val_loader
    else:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=True)
        print(len(dataset))
        return data_loader

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    train_loader, val_loader = generate_data_loader(root_dir="data_dummy",
                                        annotation_file="data_annotations.csv", 
                                        img_size=28, 
                                        batch_size=16)
    
    for i, (images, labels) in enumerate(val_loader):
        #imshow(torchvision.utils.make_grid(images))
        if i>10:
            break
    
    # get some random training images
    #dataiter = iter(data_loader)
    #images, labels = dataiter.next()
