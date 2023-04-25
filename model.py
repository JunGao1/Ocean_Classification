import torch
import torch.nn as nn
import torch.nn.functional as f

class PixelShuffle_down(nn.Module):
    def __init__(self,out_ch,in_ch,kernel):
        super(PixelShuffle_down, self).__init__()
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.kernel = kernel
        kernel_data = torch.zeros((self.out_ch,self.in_ch*self.kernel*self.kernel))
        
        for i in range(self.out_ch):
            kernel_data[i,i] = 1.0
        kernel_data = kernel_data.reshape((self.out_ch,self.in_ch,self.kernel,self.kernel))
        self.conv_weight = nn.Parameter(kernel_data,requires_grad=False)
 
    def forward(self, x):
        out = f.conv2d(x,self.conv_weight,stride=self.kernel,padding=0)
        return out

class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()
        self.down = PixelShuffle_down(64,1,8)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv6 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256,2,bias=False)
        self.dropout = nn.Dropout()
    def forward(self, x):
        x = self.down(x)
        x = self.relu(self.bn1(self.conv1(x)))
        # x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        # x = self.dropout(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.relu(self.bn5(self.conv5(x)))
        # x = self.dropout(x)
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Net_normal(nn.Module):
    def __init__(self,):
        super(Net_normal, self).__init__()
        # self.down = PixelShuffle_down(64,1,8)
        self.relu = nn.ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()
        self.bn_down1 = nn.BatchNorm2d(64)
        self.bn_down2 = nn.BatchNorm2d(128)
        self.bn_down3 = nn.BatchNorm2d(256)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.down1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=2,padding=1,bias = False)
        self.down2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1,bias = False)
        self.down3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1,bias = False)
        self.conv1 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv6 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256,2,bias=False)
        self.dropout = nn.Dropout()
    def forward(self, x):
        x = self.relu(self.bn_down1(self.down1(x)))
        x = self.relu(self.bn_down2(self.down2(x)))
        x = self.relu(self.bn_down3(self.down3(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
