import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch
import torch.nn as nn
import math

class CBAM(nn.Module):
    def __init__(self,in_channel,reduction=16,kernel_size=3):
        super(CBAM, self).__init__()
        #channel
        self.max_pool=nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        #spatial
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=kernel_size ,stride=1,padding=kernel_size//2,bias=False)

    def forward(self,x):
        #channel
        maxout=self.max_pool(x)
        maxout=self.mlp(maxout.view(maxout.size(0),-1))
        avgout=self.avg_pool(x)
        avgout=self.mlp(avgout.view(avgout.size(0),-1))
        channel_out=self.sigmoid(maxout+avgout)
        channel_out=channel_out.view(x.size(0),x.size(1),1,1)
        channel_out=channel_out*x
        #spatial
        max_out,_=torch.max(channel_out,dim=1,keepdim=True)
        mean_out=torch.mean(channel_out,dim=1,keepdim=True)
        out=torch.cat((max_out,mean_out),dim=1)
        out=self.sigmoid(self.conv(out))
        out=out*channel_out
        return out

class ResBlock(nn.Module):
    def __init__(self, channel1,channel2,channel3,kernel_size,padding):
        super(ResBlock, self).__init__()
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(channel1, channel2, kernel_size=kernel_size, padding=kernel_size//2)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(channel2, channel3, kernel_size=kernel_size, padding=kernel_size//2))
        )
    def forward(self, x):
        return x + self.module(x)

class HPSRnet(nn.Module):
    def __init__(self, args):
        super(HPSRnet, self).__init__()
        self.channel1 = args.channel1
        self.channel2 = args.channel2
        self.channel3 = args.channel3
        body3 = [ResBlock(self.channel1, self.channel1, self.channel3, kernel_size=3, padding=1) for _ in range(5)]
        body5 = [ResBlock(self.channel1, self.channel1, self.channel3, kernel_size=5, padding=2) for _ in range(5)]
        body7 = [ResBlock(self.channel1, self.channel1, self.channel3, kernel_size=7, padding=3) for _ in range(5)]
        self.head = nn.Sequential(weight_norm(nn.Conv2d(1, self.channel1, kernel_size=3, padding=1)))
        self.body3 = nn.Sequential(*body3)
        self.body5 = nn.Sequential(*body5)
        self.body7 = nn.Sequential(*body7)
        self.tail = nn.Sequential(weight_norm(nn.Conv2d(self.channel3*3, 1* (args.scale ** 2), kernel_size=3, padding=1)),
                                  nn.PixelShuffle(args.scale))
        self.skip = nn.Sequential(weight_norm(nn.Conv2d(1, 1 * (args.scale ** 2), kernel_size=5, padding=2)),
                                  nn.PixelShuffle(args.scale))
        self.dem = ResBlock(1,8,8)
        self.add = ResBlock(1,32,16)
        self.end = ResBlock(48,1,16)
        self.relu = nn.ReLU(inplace=True)
        self.CBAM = CBAM(48,12,kernel_size=7)

    def forward(self, x,y,z,pre,tmp):
        y = self.relu(self.dem(y))
        z = self.relu(self.dem(z))
        s = self.skip(x)
        x = self.head(x)
        x3 = self.body3(x)
        x5 = self.body5(x)
        x7 = self.body7(x)
        x = torch.cat([x3,x5,x7], dim=1)
        x = self.tail(x)
        x += s
        x = self.add(x)
        x = self.relu(x)
        x = torch.cat([x,y,z], dim=1)
        x = self.CBAM(x)
        x = self.end(x)
        return x