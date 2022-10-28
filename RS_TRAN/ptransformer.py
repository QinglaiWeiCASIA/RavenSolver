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

        
        if args.big:

            self.low_dim = 512
            self.name = 'Big_RS-Transformer_on_pgm'
        else:

            self.low_dim = 256
            self.name = 'RS-Transformer_on_pgm'
            
        if args.dropout:
            _dropout = 0.1
        else:
            _dropout = 0
         
            
        self.vit = nn.Sequential(ViT(image_size = 80, 
                                   patch_size = 20,  
                                   dim = self.low_dim,
                                   depth = 6, 
                                   heads = 8, 
                                   mlp_dim = self.low_dim,
                                   channels = 1, 
                                   dim_head = 32, 
                                   dropout = _dropout, 
                                   emb_dropout = _dropout),
                                 Rearrange('(b n) s d -> b n s d', n = 16)
                                 )
        
        
        
        self.g_function = nn.Sequential(
            Rearrange('b s w d -> (b s w) d'),

            Bottleneck_judge(3*self.low_dim, int(self.low_dim*0.75), 3*self.low_dim),#10,10
            
            Bottleneck_judge(3*self.low_dim, int(self.low_dim*0.75), 3*self.low_dim),#10,10
            
            Bottleneck_judge(3*self.low_dim, int(self.low_dim*0.75), self.low_dim),#10,10
            
            Rearrange('(b s w) d -> b s w d', s = 10, w = 16))
        

        
        self.high_law = nn.Sequential(
                                        Rearrange('b s w n d-> (b s w) n d'),
                                        Clstransformer(words = 6, dim = self.low_dim, depth = 6, heads = 8, dim_head = 32, mlp_dim = self.low_dim, dropout = 0.1),
                                        Rearrange('(b s w) n d-> b s w n d', s = 8, w = 16))

        self.tajador = nn.Sequential(Rearrange('b s w n d -> (b s w n) d'),
                            Bottleneck_judge(self.low_dim, self.low_dim),
                            Rearrange('(b s w n) d -> b s (w n d)', s = 8, w = 16, n = 6),
                            Mean(dim = 2))
        
        
        
        
        
    def cat(self, x):
        
        b, n, s, c = x.shape
        #x  b, 16, 16 c
        assert s == 16 and n == 16
        
        r_law = [None]*10
        
        r_law[0] = torch.cat((x[:,0], x[:,1], x[:,2]), dim = -1)#b, 16, 3*d
        r_law[1] = torch.cat((x[:,3], x[:,4], x[:,5]), dim = -1)#b, 16, 3*d
        
        
        for i in range(8):
    
            r_law[i+2] = torch.cat((x[:,6], x[:,7], x[:,i+8]), dim = -1)#b, 16, 3*d
            
        return torch.stack(r_law, dim = 1) 
    
    def c_cat(self, x):
        
        b, n, s, c = x.shape
        #x  b, 16, 64, 20, 20
        assert s == 16 and n == 16
        
        r_law = [None]*10
        
        r_law[0] = torch.cat((x[:,0], x[:,3], x[:,6]), dim = -1)#b, 16, 3*d
        r_law[1] = torch.cat((x[:,1], x[:,4], x[:,7]), dim = -1)#b, 16, 3*d
        
        
        for i in range(8):
    
            r_law[i+2] = torch.cat((x[:,2], x[:,5], x[:,i+8]), dim = -1)#b, 16, 3*d
            
        return torch.stack(r_law, dim = 1) # b, 10, 16, 3*d
    
    
    def combine(self, r_law):
        r_combine = [None]*8
        assert r_law.shape[1] == 10 and r_law.shape[2] == 16 and r_law.shape[3] == self.low_dim
            
        for i in range(8):
            r_combine[i] = torch.stack((r_law[:,0], r_law[:,1], r_law[:,2+i]), dim = 2)#b, 16 ,3 ,d, 
            
        
            
        return torch.stack(r_combine, dim = 1)#b ,8, 16, 3, d
    
    def rc_add(self, *args):
        assert args[0].shape[2] == 16 and args[0].shape[1] == 8 and args[0].shape[3] == 3
        return torch.cat(args, dim = -2)

    
    
    def forward(self, x):
        b, n, h, w = x.shape
        x = x.reshape(b*n, -1, h, w)
        
        x = self.vit(x)
        
        cat_row = self.cat(x)
        cat_cow = self.c_cat(x)
        
        r_law = self.g_function(cat_row)
        c_law = self.g_function(cat_cow)
        
        rc_law = self.combine(r_law)
        cc_law = self.combine(c_law)
        

        law = self.rc_add(rc_law, cc_law)

        high = self.high_law(law)

        return self.tajador(high).reshape(b,8), torch.empty(0).to(device)

    def choose(self, *out):
   
        r = out[0]

        return F.softmax(r, dim = -1)
    

    
    
    def loss_function(self, *out, target, temperature = 1):
    
        r = out[0]

        loss_1 = F.cross_entropy(r / temperature, target)

        return loss_1
        
def transpose(x):
    return x.transpose(-2, -1).contiguous()
    
def mul_dot(a, b):
    
    assert a.dim() == b.dim() and a.dim() == 3 and b.shape[1] == 1  and a.shape[1] == 8, 'dim error'

    return (a@transpose(b)).squeeze(-1)
    
    
def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]        

 
if __name__ == '__main__':
    from torchsummary  import summary
    model = reasoning().cuda()
    summary(model.cuda(), (16,80,80))
    

