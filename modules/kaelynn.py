import torch
import torch.nn as nn

class KAELYNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.25)

        self.lin1 = nn.Linear(200704, 16)
        self.lin2 = nn.Linear(16, 32)
        self.last_lin = nn.Linear(32, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.drop(x)

        x = x.view(x.size(0), -1) 

        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.last_lin(x)
        return x

if __name__ == '__main__':
    net = KAELYNN()
    x = torch.randn(2,3,224,224)
    print(net(x).shape)