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
    def __init__(self, args):
        super(reasoning,self).__init__()
        self.name = 'pcnn_on_pgm'
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
                'mlp': 0.5,
            }
        else:
            _dropout = {
                'high': 0.,
                'mid': 0.,
                'low': 0.,
                'mlp': 0.,
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
        
        

        self.judgement = nn.Sequential(
                                Bottleneck_judge(in_places=2*(self.out_high_dim + self.out_mid_dim + self.out_low_dim), 
                                               hidden_places=2*(self.out_high_dim + self.out_mid_dim + self.out_low_dim), 
                                               out_places=2*(self.out_high_dim + self.out_mid_dim + self.out_low_dim)),
                                nn.ReLU(),
                                Bottleneck_judge(in_places=2*(self.out_high_dim + self.out_mid_dim + self.out_low_dim), 
                                               hidden_places=2*(self.out_high_dim + self.out_mid_dim + self.out_low_dim), 
                                               out_places=2*(self.out_high_dim + self.out_mid_dim + self.out_low_dim))


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
    
    def c_cat(self, x):#b,16,...   b,10,...
        
        b, s, c, h, w = x.shape
        #x  b, 16, 64, 20, 20
        assert s == 16
        
        r_law = [None]*10
        
        r_law[0] = torch.stack((x[:,0], x[:,3], x[:,6]), dim = 1).reshape(b, 3*c, h, w)#b, 3c, h, w
        r_law[1] = torch.stack((x[:,1], x[:,4], x[:,7]), dim = 1).reshape(b, 3*c, h, w)#b, 3c, h, w
        
        
        for i in range(8):
    
            r_law[i+2] = torch.stack((x[:,2],x[:,5],x[:,i+8]), dim = 1).reshape(b, 3*c, h, w)#b, 3, f, dim   
            
        return torch.stack(r_law, dim = 1) # b, 10, 3*c, h, w
    
    
    def combine(self, r_law, r_answer):
        r_combine = [None]*20
            
        for i in range(8):
            r_combine[i] = torch.cat((r_law[:,0], r_law[:,i+2]), dim = -1)#b, f, 2*law_dim     i = 7             1c
            
        for i in range(8):
            r_combine[i+8] = torch.cat((r_law[:,1], r_law[:,i+2]), dim = -1)#b, f, 2*law_dim     i = 15          2c
            
        
        r_combine[16] = torch.cat((r_law[:,0], r_law[:,1]), dim = -1)# b, f, 2*law_dim                           12
        
        r_combine[17] = torch.cat((r_law[:,1], r_law[:,0]), dim = -1)# b, f, 2*dim                                    21
        
        
        if self.training == True:

            r_combine[18] = torch.cat((r_answer, r_law[:,0]), dim = -1)# b, f, 2*dim                              c1
        
            r_combine[19] = torch.cat((r_answer, r_law[:,1]), dim = -1)# b, f, 2*dim                              c2
                
            
            

            r_combine = (torch.stack(r_combine, dim = 1))# b, 20, hw, 2*law_dim
                
        else:
            
            r_combine[18:20] = []
           
            r_combine = (torch.stack(r_combine, dim = 1))# b, 18, hw, 2*law_dim 
            
            
        return (r_combine)
    
    
    def add_vector(self, r,c):
        assert r.shape[1] == 20, c.shape[1] == 20
        
        out = [None]*20
        
        
        for i in range(0,4):
            out[i] = torch.cat((r[:,16:17] ,c[:,16+i:17+i]), dim = -1)#0-3
            
        for i in range(0,4):
            out[i+4] = torch.cat((r[:,17:18] , c[:,16+i:17+i]), dim = -1)#4-7
            
        for i in range(0,4):
            out[i+8] = torch.cat((r[:,18:19] , c[:,16+i:17+i]), dim = -1)#8-11
            
        for i in range(0,4):
            out[i+12] = torch.cat((r[:,19:20] , c[:,16+i:17+i]), dim = -1) #12-15
            
        out[16] = torch.cat((r[:,:8] , c[:,:8]), dim = -1)  #16-23
        
        out[17] = torch.cat((r[:,:8] , c[:,8:16]), dim = -1)  #24-31
        
        out[18] = torch.cat((r[:,8:16] , c[:,:8]), dim = -1)  #32-39
        
        out[19] = torch.cat((r[:,8:16] , c[:,8:16]), dim = -1)#40-47
        
        return torch.cat(out, dim = 1)
    
    def splic_vector(self, in_):
        assert in_.shape[1] == 48
        
        
        
        out = [None]*20
        for i in range(16):
            out[i] = in_[:,i:i+1]
            
        out[16] = in_[:,16:24]
        out[17] = in_[:,24:32]
        out[18] = in_[:,32:40]
        out[19] = in_[:,40:48]
        
    
        
        return out
    
    
    def add_vector_choose(self, r,c):
        assert (r.shape[1] == 18  and  c.shape[1] == 18) or (r.shape[1] == 20  and  c.shape[1] == 20)
        
        out = [None]*8
        
        
        for i in range(0,2):
            out[i] = torch.cat((r[:,16:17] , c[:,16+i:17+i]), dim = -1)#0-1
            
        for i in range(0,2):
            out[i+2] = torch.cat((r[:,17:18] , c[:,16+i:17+i]), dim = -1)#2-3
            
            
        out[4] = torch.cat((r[:,:8] , c[:,:8] ), dim = -1) #16-23
        
        out[5] = torch.cat((r[:,:8] , c[:,8:16]  ), dim = -1)#24-31
        
        out[6] = torch.cat((r[:,8:16] , c[:,:8]  ), dim = -1)#32-39
        
        out[7] = torch.cat((r[:,8:16] , c[:,8:16]), dim = -1)#40-47
        
        return torch.cat(out, dim = 1)
        
    def splic_vector_choose(self, in_):
        assert in_.shape[1] == 36
        
        
        
        out = [None]*8
        for i in range(4):
            out[i] = in_[:,i:i+1]
            
        out[4] = in_[:,4:12]
        out[5] = in_[:,12:20]
        out[6] = in_[:,20:28]
        out[7] = in_[:,28:36]
        
    
        
        return out
    
    def forward(self, x, y):
        b, n, h, w = x.shape
        x = x.reshape(b*n, -1, h, w)
        
        input_features_high = self.perception_net_high(x)
        input_features_mid = self.perception_net_mid(input_features_high)
        input_features_low = self.perception_net_low(input_features_mid)
        
        input_features_high = input_features_high.reshape(b, n, input_features_high.shape[-3], input_features_high.shape[-2], input_features_high.shape[-1])
        
        input_features_mid = input_features_mid.reshape(b, n, input_features_mid.shape[-3], input_features_mid.shape[-2], input_features_mid.shape[-1])
        
        input_features_low = input_features_low.reshape(b, n, input_features_low.shape[-3], input_features_low.shape[-2], input_features_low.shape[-1])
        
        
        
        
        cat_row_high = self.cat(input_features_high)# b, 10, 3*c, h, w
        
        cat_row_mid = self.cat(input_features_mid)# b, 10, 3*c, h, w
        
        cat_row_low = self.cat(input_features_low)# b, 10, 3*c, h, w

        
        cat_cow_high = self.c_cat(input_features_high)# b, 10, 3*c, h, w
        
        cat_cow_mid = self.c_cat(input_features_mid)# b, 10, 3*c, h, w
        
        cat_cow_low = self.c_cat(input_features_low)# b, 10, 3*c, h, w
        
        
        
        
        
        #s = 10
        r_law_high = self.g_function_high(cat_row_high)# b, s, hw, c
        
        r_law_mid = self.g_function_mid(cat_row_mid)# b, s, hw, c
        
        r_law_low = self.g_function_low(cat_row_low)# b, s, hw, c
        
        
        c_law_high = self.g_function_high(cat_cow_high)# b, s, hw, c
        
        c_law_mid = self.g_function_mid(cat_cow_mid)# b, s, hw, c
        
        c_law_low = self.g_function_low(cat_cow_low)# b, s, hw, c
        

        
        
        if self.training == True:
            y = F.one_hot(y,8)[:,:,None, None, None]
        
            r_answer_high = self.get_answer(r_law_high, y)
            r_answer_mid = self.get_answer(r_law_mid, y)
            r_answer_low = self.get_answer(r_law_low, y)
            
            c_answer_high = self.get_answer(c_law_high, y)
            c_answer_mid = self.get_answer(c_law_mid, y)
            c_answer_low = self.get_answer(c_law_low, y)
            
            
            
        else:
            r_answer_high = None
            r_answer_mid = None
            r_answer_low = None
            
            c_answer_high = None
            c_answer_mid = None
            c_answer_low = None
            
        
        
        
        
        
        #s = 20
        rc_law_high = self.combine(r_law_high, r_answer_high)#b, s, hw, 2*c
        
        rc_law_mid = self.combine(r_law_mid, r_answer_mid)#b, s, hw, 2*c
        
        rc_law_low = self.combine(r_law_low, r_answer_low)#b, s, hw, 2*c
        
        
        
        cc_law_high = self.combine(c_law_high, c_answer_high)#b, s, hw, 2*c
        
        cc_law_mid = self.combine(c_law_mid, c_answer_mid)#b, s, hw, 2*c
        
        cc_law_low = self.combine(c_law_low, c_answer_low)#b, s, hw, 2*c
        

        

        

        high_r = self.high_law(rc_law_high)#b, s, hw, law_dim -> b, s, law_dim
        
        mid_r = self.mid_law(rc_law_mid)#b, s, hw, law_dim -> b, s, law_dim
        
        low_r = self.low_law(rc_law_low)#b, s, hw, law_dim -> b, s, law_dim
        
        # print(high_r.shape)
        
        high_c = self.high_law(cc_law_high)#b, s, hw, law_dim -> b, s, law_dim
        
        mid_c = self.mid_law(cc_law_mid)#b, s, hw, law_dim -> b, s, law_dim
        
        low_c = self.low_law(cc_law_low)#b, s, hw, law_dim -> b, s, law_dim
        
        

        
        

        

        
        return high_r.reshape(b,-1,self.out_high_dim), mid_r.reshape(b,-1,self.out_mid_dim), low_r.reshape(b,-1,self.out_low_dim),\
            high_c.reshape(b,-1,self.out_high_dim), mid_c.reshape(b,-1,self.out_mid_dim), low_c.reshape(b,-1,self.out_low_dim)
            
    
    def get_answer(self, r_law, y):
        
        return (y*r_law[:, 2:10]).sum(dim = 1)
    
    def depth_choose(self, title, choise):

        return mul_dot(choise,title) 
    
        
    def choose(self, *out):
        
        # print(r[0].shape)
        
        
        r = torch.cat((out[:3]), dim = -1)
        
        c = torch.cat((out[3:]), dim = -1)

        state = self.judgement.training

        self.judgement.eval()
        
        out = self.add_vector_choose(r, c)
        out_reverse = self.add_vector_choose(c, r)
        
        
        out = self.splic_vector_choose(self.judgement(out))
        out_reverse = self.splic_vector_choose(self.judgement(out_reverse))
        
        out = normalize(*out)
        out_reverse = normalize(*out_reverse)
        
        
        assert len(out) == 8
        
        loss_1 = 0
        loss_2 = 0
        loss_3 = 0
        loss_4 = 0
        
        for i in range(4):
        
            loss_1 += self.depth_choose(out[i], out[4]) + self.depth_choose(out_reverse[i], out_reverse[4])
            loss_2 += self.depth_choose(out[i], out[5]) + self.depth_choose(out_reverse[i], out_reverse[5])
            loss_3 += self.depth_choose(out[i], out[6]) + self.depth_choose(out_reverse[i], out_reverse[6])
            loss_4 += self.depth_choose(out[i], out[7]) + self.depth_choose(out_reverse[i], out_reverse[7])
        

        self.judgement.training = state
        
        return F.softmax(loss_1+ loss_2+ loss_3+ loss_4, dim = -1)
    
    
    
        
        
    def loss_baby(self, title, choise, target, temperature):



        return F.cross_entropy(mul_dot(choise, title) / temperature, target)

    
    
    def loss_function(self, *out, target, temperature):
        
        
        r = torch.cat((out[:3]), dim = -1)
        
        c = torch.cat((out[3:]), dim = -1)
        
        out = self.add_vector(r, c)
        print(out.shape)
        
        out_reverse = self.add_vector(c, r)
        


        out = self.splic_vector(self.judgement(out))
        out_reverse = self.splic_vector(self.judgement(out_reverse))
        
        
        out = normalize(*out)
        out_reverse = normalize(*out_reverse)
        assert len(out) == 20
        
        
        loss_1 = 0
        loss_2 = 0
        loss_3 = 0
        loss_4 = 0
        
        for i in range(16):
            loss_1 += self.loss_baby(out[i], out[16], target, temperature) + self.loss_baby(out_reverse[i], out_reverse[16], target, temperature)
            loss_2 += self.loss_baby(out[i], out[17], target, temperature) + self.loss_baby(out_reverse[i], out_reverse[17], target, temperature)
            loss_3 += self.loss_baby(out[i], out[18], target, temperature) + self.loss_baby(out_reverse[i], out_reverse[18], target, temperature)
            loss_4 += self.loss_baby(out[i], out[19], target, temperature) + self.loss_baby(out_reverse[i], out_reverse[19], target, temperature)
       
        


        return loss_1 + loss_2 + loss_3 + loss_4
        
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
    x = torch.randint(0,8,(5,16,80,80)).to(device).float()
    y = torch.randint(0,8,(5,)).to(device).long()
    
    model = reasoning().to(device)
    
    # summary(model.cuda(), )
    
    r = model(x, y)
    
    m = model.choose(*r)
    
    loss = model.loss_function(*r, target = y)
    
    loss.backward()
    
    model.eval()
    
    r_ = model(x, 0)
