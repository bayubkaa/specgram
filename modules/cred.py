import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class block_CNN(nn.Module):
    def __init__(self, block_index):
        super().__init__()
        self.bn_block1 = nn.BatchNorm2d(8)
        self.conv_block1 = nn.Conv2d(8, 8, kernel_size=9-2, padding='same')

        self.bn_block2 = nn.BatchNorm2d(16)
        self.conv_block2 = nn.Conv2d(16, 16, kernel_size=5-2, padding='same')

        self.block_index = block_index

        if self.block_index==1:
            self.bn = self.bn_block1
            self.conv = self.conv_block1
        else:
            self.bn = self.bn_block2
            self.conv = self.conv_block2
        self.relu = nn.ReLU()
    def forward(self, x): 
        """
        Returns CNN residual blocks
        """
        layer_1 = self.bn(x) 
        act_1 = self.relu(layer_1) 

        conv_1 = self.conv(act_1)  
        
        layer_2 = self.bn(conv_1) 
        act_2 = self.relu(layer_2) 
    
        conv_2 = self.conv(act_2) 
        return(conv_2) 


class CRED(nn.Module):
    def __init__(self, batch_size, num_classes=2, size=224):
        super(CRED, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=9, stride=(2,2))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=(2,2))
        self.relu = nn.ReLU()
        #self.fc_mid = nn.Linear(-1,)

        if size==224:
            lstm_size1 = 832
            bn_uni_size = 52
            last_lin_size = 104
        else:
            lstm_size1 = 112
            bn_uni_size = 34
            last_lin_size = 68

        self.res_lstm1 = torch.nn.LSTM(input_size=lstm_size1, hidden_size=64, dropout=0.7, bidirectional=True, num_layers=2)
 
        self.uni_lstm = torch.nn.LSTM(input_size=128, hidden_size=64, dropout=0.8)
        self.bn_uni = nn.BatchNorm1d(bn_uni_size)

        self.linear_TD1 = nn.Linear(64, 64)
        self.do_TD = nn.Dropout(0.8)
        self.bn_TD = nn.BatchNorm1d(bn_uni_size)
        self.time_distributed1 = TimeDistributed(self.linear_TD1)

        self.linear_TD2 = nn.Linear(64, 2)
        self.time_distributed2 = TimeDistributed(self.linear_TD2)

        self.last_linear = nn.Linear(last_lin_size, 2)
        
        self.block_cnn1 = block_CNN(1)
        self.block_cnn2 = block_CNN(2)
    
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.relu(x)
        #x += self.block_cnn1(x)
        x = x.clone() + self.block_cnn1(x)
        x = self.conv2(x)
        x = self.relu(x)
        #x += self.block_cnn2(x)
        x = x.clone() + self.block_cnn2(x)
        
        x = x.view(x.size(0), x.size(2), -1) 
        
        res_lstm1, _ = self.res_lstm1(x)
        
        out_uni, _ = self.uni_lstm(res_lstm1)
        x = self.bn_uni(out_uni)

        x = self.time_distributed1(x)
        x = self.bn_TD(x)
        x = self.do_TD(x)
        x = self.time_distributed2(x)

        x = x.view(x.size(0), -1) 
        x = self.last_linear(x)
        return x

#68 104
if __name__ == "__main__":
    size = (151,41)
    batch_size = 2
    x = torch.randn(batch_size,3,size[0],size[1], device='cuda')
    model = CRED(batch_size=batch_size,size=size[0])
    model.to('cuda')

    y = model(x)
    print(y)
    print(y.size())