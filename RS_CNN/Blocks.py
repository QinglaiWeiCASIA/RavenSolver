import torch
import torch.nn as nn
def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if downsample is None:
            self.downsample = nn.Identity()
        else:
            self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x.contiguous())))
        out = self.relu(self.downsample(x.contiguous()) + self.bn2(self.conv2(out.contiguous())))

        return out


class ResBlock1x1(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock1x1, self).__init__()
        self.conv1 = conv1x1(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv1x1(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x.contiguous())))
        out = self.relu(x.contiguous() + self.bn2(self.conv2(out.contiguous())))

        return out
    
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)
    
class Mean(nn.Module):
    def __init__(self, dim, keepdim = False):
        super(Reshape, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return x.mean(dim = self.dim, keepdim = self.keepdim)
    
    
class Bottleneck_judge(nn.Module):
    def __init__(self, in_places, hidden_places, out_places = 1,  dropout = 0.1, last_dropout = 0.5):
        super(Bottleneck_judge,self).__init__()

        self.out_places = out_places

        self.bottleneck = nn.Sequential(
            nn.Linear(in_places, hidden_places),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_places),
            nn.Linear(hidden_places, hidden_places),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_places),
            nn.Linear(hidden_places, out_places)
        )

        if in_places != out_places:
            self.downsample = nn.Sequential(
                nn.Linear(in_places, out_places)
            )
            
        else:
            self.downsample = nn.Identity()
            

    def forward(self, x):
        b,n,d = x.shape
        
        x = x.reshape(-1, d)

        out = self.bottleneck(x)
        
        residual = self.downsample(x)
        
        out += residual
        
        return out.reshape(b,n,self.out_places)


    

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1, downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        


        if self.downsampling:
            residual = self.downsample(x.contiguous())
            


        out += residual
        out = self.relu(out)
        return out