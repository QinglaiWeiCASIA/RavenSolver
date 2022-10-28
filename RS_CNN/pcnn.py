#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:26:31 2021

@author: yuanbeiming
"""
import torch
import torch.nn as nn

import torch.nn.functional as F


if __name__ == '__main__':
    from Blocks import *
else:  
    from .Blocks import *


from einops.layers.torch import Rearrange




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    
class reasoning(nn.Module):
    def __init__(self, args ):
        super(reasoning,self).__init__()
        self.name = 'miracle_pcnn'
        if args.big:
            self.high_dim, self.high_dim0 = 128, 64
            self.mid_dim, self.mid_dim0 = 256, 128
            self.low_dim, self.low_dim0 = 512, 256
        else:
            self.high_dim, self.high_dim0 = 64, 32
            self.mid_dim, self.mid_dim0 = 128, 64
            self.low_dim, self.low_dim0 = 256, 128
            
        
            
        if args.dropout:
            _dropout = {
                'high': 0.1,
                'mid': 0.1,
                'low': 0.1,
             
            }
        else:
            _dropout = {
                'high': 0.,
                'mid': 0.,
                'low': 0.,
               
            }
        
        
        if args.more_big:
        
            self.out_high_dim = self.low_dim
            
            self.out_mid_dim = self.low_dim
            
            self.out_low_dim = self.low_dim
            
        else:
            
            self.out_high_dim = self.mid_dim
            
            self.out_mid_dim = self.mid_dim
            
            self.out_low_dim = self.mid_dim

        
            
        
            
        self.perception_net_high = nn.Sequential(
            nn.Conv2d(1, self.high_dim0, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.high_dim0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(_dropout['high']),
            nn.Conv2d(self.high_dim0, self.high_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.high_dim),
            nn.ReLU(inplace=True)
            )

        self.perception_net_mid = nn.Sequential(
            nn.Conv2d(self.high_dim, self.mid_dim0, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_dim0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(_dropout['mid']),
            nn.Conv2d(self.mid_dim0, self.mid_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_dim),
            nn.ReLU(inplace=True)
            )

        self.perception_net_low = nn.Sequential(
            nn.Conv2d(self.mid_dim, self.low_dim0, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.low_dim0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(_dropout['low']),
            nn.Conv2d(self.low_dim0, self.low_dim, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(self.low_dim),
            nn.ReLU(inplace=True)
            )
        
        
        self.g_function_high = nn.Sequential(Reshape(shape=(-1, 3 * self.high_dim, 20, 20)),# b, 10, 3*c, h, w ->  b*10, 3*c, h, w
                                             conv3x3(3 * self.high_dim, self.high_dim),
                                             ResBlock(self.high_dim, self.high_dim),
                                             ResBlock(self.high_dim, self.high_dim),
                                             Reshape(shape=(-1, 10, self.high_dim, 20, 20)),
                                             Rearrange('b s c h w -> b s h w c'))
        self.g_function_mid = nn.Sequential(Reshape(shape=(-1, 3 * self.mid_dim, 5, 5)),
                                            conv3x3(3 * self.mid_dim, self.mid_dim),
                                            ResBlock(self.mid_dim, self.mid_dim),
                                            ResBlock(self.mid_dim, self.mid_dim),
                                            Reshape(shape=(-1, 10, self.mid_dim, 5, 5)),
                                            Rearrange('b s c h w -> b s h w c'))
        self.g_function_low = nn.Sequential(Reshape(shape=(-1, 3 * self.low_dim, 1, 1)),
                                            conv1x1(3 * self.low_dim, self.low_dim),
                                            ResBlock1x1(self.low_dim, self.low_dim),
                                            ResBlock1x1(self.low_dim, self.low_dim),
                                            Reshape(shape=(-1, 10, self.low_dim, 1, 1)),
                                            Rearrange('b s c h w -> b s h w c'))
        
        
        self.high_law = nn.Sequential(
            
            Rearrange('b s h w c -> (b s) c h w'),

            Bottleneck(self.high_dim*2, int(self.high_dim*0.5), stride = 2, downsampling =True, expansion = 4),#10,10
            
            Bottleneck(self.high_dim*2, int(self.high_dim*0.5), stride = 2, downsampling =True, expansion = 4),#5*5
            
            Bottleneck(self.high_dim*2, int(self.high_dim*0.5), stride = 2, downsampling =True, expansion = 4),#3*3
            
            nn.Conv2d(self.high_dim*2, self.out_high_dim, kernel_size=3, stride=1, padding=0),
            
            Rearrange('b c h w -> b (c h w)')
            
        )
        
        self.mid_law = nn.Sequential(
            
            Rearrange('b s h w c -> (b s) c h w'),

            Bottleneck(self.mid_dim*2, int(self.mid_dim*0.5), stride = 2, downsampling =True, expansion = 4),#3*3
            
            Bottleneck(self.mid_dim*2, int(self.mid_dim*0.5), stride = 2, downsampling =True, expansion = 4),#2*2
            
            nn.Conv2d(self.mid_dim*2, self.out_mid_dim, kernel_size=2, stride=1, padding=0),
            
            Rearrange('b c h w -> b (c h w)')
        )
        
        self.low_law = nn.Sequential(
            
            Rearrange('b s h w c -> (b s) c h w'),

            nn.Conv2d(self.low_dim*2, self.out_low_dim, kernel_size=3, stride=1, padding=1),
            
            Rearrange('b c h w -> b (c h w)')
        )
        
        
    
        
    def cat(self, x):#b,16,...   b,10,...
        
        b, s, c, h, w = x.shape
        #x  b, 16, 64, 20, 20
        assert s == 16
        
        r_law = [None]*10
        
        r_law[0] = (x[:,:3]).reshape(b, 3*c, h, w)#b, 3, f, dim  
        r_law[1] = (x[:,3:6]).reshape(b, 3*c, h, w)#b, 3, f, dim  
        
        
        for i in range(8):
    
            r_law[i+2] = (torch.cat((x[:,6:8],x[:,i+8:i+9]), dim = 1)).reshape(b, 3*c, h, w)#b, 3, f, dim   
            
        return torch.stack(r_law, dim = 1) # b, 10, 3*c, h, w
    
    
    def combine(self, r_law, r_answer):
        r_combine = [None]*20
            
        for i in range(8):
            r_combine[i] = torch.cat((r_law[:,0], r_law[:,i+2]), dim = -1)#b, f, 2*law_dim     i = 7
            
        for i in range(8):
            r_combine[i+8] = torch.cat((r_law[:,1], r_law[:,i+2]), dim = -1)#b, f, 2*law_dim     i = 15
            
        
        r_combine[16] = torch.cat((r_law[:,0], r_law[:,1]), dim = -1)# b, f, 2*law_dim
        
        r_combine[17] = torch.cat((r_law[:,1], r_law[:,0]), dim = -1)# b, f, 2*dim  
        
        
        if self.training == True:
            
            
        
            
            
            
            r_combine[18] = torch.cat((r_answer, r_law[:,0]), dim = -1)# b, f, 2*dim
        
            r_combine[19] = torch.cat((r_answer, r_law[:,1]), dim = -1)# b, f, 2*dim
                
            
            

            r_combine = (torch.stack(r_combine, dim = 1))# b, 20, hw, 2*law_dim
                
        else:
            
            
            
            r_combine[18:20] = []
           
            r_combine = (torch.stack(r_combine, dim = 1))# b, 17, hw, 2*law_dim 
            
            
        return (r_combine)
    
    
    def forward(self, x, y = None):
        b, n, h, w = x.shape
        x = x.reshape(b*n, -1, h, w)
        y = torch.randint(8,(2,)).cuda()
        
        input_features_high = self.perception_net_high(x)
        input_features_mid = self.perception_net_mid(input_features_high)
        input_features_low = self.perception_net_low(input_features_mid)
        
        input_features_high = input_features_high.reshape(b, n, input_features_high.shape[-3], input_features_high.shape[-2], input_features_high.shape[-1])
        
        input_features_mid = input_features_mid.reshape(b, n, input_features_mid.shape[-3], input_features_mid.shape[-2], input_features_mid.shape[-1])
        
        input_features_low = input_features_low.reshape(b, n, input_features_low.shape[-3], input_features_low.shape[-2], input_features_low.shape[-1])
        
        
        cat_raw_high = self.cat(input_features_high)# b, 10, 3*c, h, w
        
        cat_row_mid = self.cat(input_features_mid)# b, 10, 3*c, h, w
        
        cat_row_low = self.cat(input_features_low)# b, 10, 3*c, h, w
        # 
        print(cat_row_mid.shape)
        
        
        #s = 10
        r_law_high = self.g_function_high(cat_raw_high)# b, s, hw, c
        
        r_law_mid = self.g_function_mid(cat_row_mid)# b, s, hw, c
        
        r_law_low = self.g_function_low(cat_row_low)# b, s, hw, c
        

        print(r_law_mid.shape)
        
        if self.training == True:
            y = F.one_hot(y,8)[:,:,None, None, None]
        
            r_answer_high = self.get_answer(r_law_high, y)
            r_answer_mid = self.get_answer(r_law_mid, y)
            r_answer_low = self.get_answer(r_law_low, y)
            
        else:
            r_answer_high = None
            r_answer_mid = None
            r_answer_low = None
            
        
        
        
        
        
        #s = 20
        c_law_high = self.combine(r_law_high, r_answer_high)#b, s, hw, 2*c
        
        c_law_mid = self.combine(r_law_mid, r_answer_mid)#b, s, hw, 2*c
        
        c_law_low = self.combine(r_law_low, r_answer_low)#b, s, hw, 2*c
        

        

        

        high_ = self.high_law(c_law_high)#b, s, hw, law_dim -> b, s, law_dim
        
        mid_ = self.mid_law(c_law_mid)#b, s, hw, law_dim -> b, s, law_dim
        
        low_ = self.low_law(c_law_low)#b, s, hw, law_dim -> b, s, law_dim
        
        

        

        
        return high_.reshape(b,-1,self.out_high_dim), mid_.reshape(b,-1,self.out_mid_dim), low_.reshape(b,-1,self.out_low_dim)
    
    def get_answer(self, r_law, y):
        
        
        
            
        return (y*r_law[:, 2:10]).sum(dim = 1)
    
    def depth_choose(self, r_combine):

        
        assert r_combine.shape[1] == 20 or r_combine.shape[1] == 18, 'error'
        
        answer_1 = r_combine[:,:8]  
        
        answer_2 = r_combine[:,8:16]
        
        # choise = (answer_1+answer_2)*0.5
        
        title = r_combine[:,16:17]#b,1,dim
        
        title_reverse = r_combine[:,17:18]
        
        
        
        
        return mul_dot(answer_1,title) + mul_dot(answer_1,title_reverse) + mul_dot(answer_2,title) + mul_dot(answer_2,title_reverse)
    
        
    def choose(self, *r):
        
        # print(r[0].shape)
        
        
        r_combine = torch.cat((r), dim = -1)
        
        r_combine = F.normalize(r_combine, dim = -1)
        
        # print(r.shape)
        
        r_combine = self.depth_choose(r_combine)
        

        
        
        return F.softmax(r_combine, dim = -1)
    
    
    
        
        
    def loss_baby(self, r_combine, target, temperature):
        

        
        assert r_combine.shape[1] == 20, 'error'
        
        answer_1 = r_combine[:,:8]
        
        answer_2 = r_combine[:,8:16]
        
        # choise = (answer_1 + answer_2)*0.5
        
        title = r_combine[:,16:17]
        
        title_reverse = r_combine[:,17:18]
        
        answer = r_combine[:,18:19]
        
        answer_reverse = r_combine[:,19:20]
        
        
        # choise,title,title_reverse,answer,answer_reverse = normalize(choise,title,title_reverse,answer,answer_reverse)
        
        loss1 = F.cross_entropy(mul_dot(answer_1,title) / temperature, target)
        
        loss2 = F.cross_entropy(mul_dot(answer_1,title_reverse) / temperature, target)
        
        loss3 = F.cross_entropy(mul_dot(answer_1,answer) / temperature, target)
        
        loss4 = F.cross_entropy(mul_dot(answer_1,answer_reverse) / temperature, target)
        
        
        
        loss5 = F.cross_entropy(mul_dot(answer_2,title) / temperature, target)
        
        loss6 = F.cross_entropy(mul_dot(answer_2,title_reverse) / temperature, target)
        
        loss7 = F.cross_entropy(mul_dot(answer_2,answer) / temperature, target)
        
        loss8 = F.cross_entropy(mul_dot(answer_2,answer_reverse) / temperature, target)
        
        
        

        return loss1 + loss2 + loss3 +  loss4 + loss5 + loss6 + loss7 +  loss8

    
    
    def loss_function(self, *r, target, temperature):
        
        
        r = torch.cat((r), dim = -1)
        
        r = F.normalize(r, dim = -1)
        


        
        loss_1 = self.loss_baby(r, target, temperature)
        


        return loss_1
        
def transpose(x):
    return x.transpose(-2, -1).contiguous()
    
def mul_dot(a, b):
    
    assert a.dim() == b.dim() and a.dim() == 3 and b.shape[1] == 1  and a.shape[1] == 8, 'error'
    
    # a@transpose(b)
    
    # print(a.shape,b.shape, (a@transpose(b)).shape)
    return (a@transpose(b)).squeeze(-1)
    
    
def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]  

 
if __name__ == '__main__':
    from torchsummary import summary
    x = torch.randint(0,8,(2,16,80,80)).to(device).float()
    y = torch.randint(0,8,(2,)).to(device).long()
    
    model = reasoning().to(device)
    summary(model,(16,80,80))
   
    model.train()
    r = model(x, y)
    
    m = model.choose(*r)
    
    loss = model.loss_function(*r, target = y)
    
    loss.backward()
    
    model.eval()
    
    r_ = model(x, 0)
