#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:26:31 2021

@author: yuanbeiming
"""
import torch
import torch.nn as nn
import numpy as np


import torch.nn.functional as F

from Blocks_clip import *


from einops.layers.torch import Rearrange


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Recat(nn.Module):
    def __init__(self):
        super(Recat, self).__init__()


    def forward(self, x):
        b ,n ,s ,d = x.shape
        
    

        return x[:, [0,1,2,3,4,5,6,7,8,
                    
                    0,3,6,1,4,7,2,5,8,
                   
                    ]].reshape(b, 6, 3, s, d)
    
class Recombine(nn.Module):
    def __init__(self):
        super(Recombine, self).__init__()
       

    def forward(self, x):
        b ,s ,m ,d = x.shape
        

        
        return x[:,:, [0,1,2,10,11,12,
                       0,1,3,10,11,13,
                       0,1,4,10,11,14,
                       0,1,5,10,11,15,
                       0,1,6,10,11,16,
                       0,1,7,10,11,17,
                       0,1,8,10,11,18,
                       0,1,9,10,11,19
                    ]].reshape(b, s, 8, 6, d)
    


    
    
    
class raven_clip(nn.Module):
    def __init__(self, args):
        super(raven_clip,self).__init__()

        self.name = 'Clip_pgm_perfect'

        size = 80
        patch = 20
        
        if args.big:
            num_head = 8
            num_depth = 6
            self.low_dim = 256
        else:
            num_head = 4
            num_depth = 3
            self.low_dim = 128
            
        if args.dropout:
            _dropout = 0.1
        else:
            _dropout = 0
        txt_data = []
        for c in range(8,14):#color
            for n in range(8,14):#number
                for p in range(8,14):#position
                    for s in range(8,14):#size
                        for t in range(8,14):#type
                            txt_data.append(np.array([ 3,  c,  4, n,  5, p,  6, s, 7, t]))
                        
        



        
        txt_data = np.array(txt_data)[:-1]
        
        assert txt_data.shape[0] == 7775
        
        txt_size = 7775
        
        
        self.dict_ = { 'EOF':0,
                      'shape':1,
                      'line':2, 
                      'color':3, 
                      'number':4, 
                      'position':5, 
                      'size':6, 
                      'type':7, 
                      'progression':8, 
                      'XOR':9, 
                      'OR':10, 
                      'AND':11, 
                      'consistent_union':12,
                      ' NA':13}
        
        self.txt_data = torch.from_numpy(txt_data[None,:,:]).long()
        
        
        assert self.txt_data.shape[1] == 7775
        
        self.txt_data.requires_grad = False
    

        self.w = int(size/patch)*int(size/patch)


        self.temperature = 1e-6
            
        self.vit = nn.Sequential(ViT(image_size = size, 
                                   patch_size = patch,  
                                   dim = self.low_dim,
                                   depth = num_depth, 
                                   heads = num_head, 
                                   mlp_dim = self.low_dim,
                                   channels = 1, 
                                   dim_head = int(self.low_dim/num_head), 
                                   dropout = _dropout, 
                                   emb_dropout = _dropout),
                                 Rearrange('(b n) s d -> b n s d', n = 9),
                                 Recat(),
                                 Rearrange('b m n s d -> b s m (n d)', s = self.w, n = 3, m = 6),
                                 

                                 )#b*16, 16, dim
        
        
        
        self.g_function = nn.Sequential(
            Rearrange('b s m d -> (b s m) d'),

            Bottleneck_judge(3*self.low_dim, int(self.low_dim*0.75), 3*self.low_dim),#10,10
            
            Bottleneck_judge(3*self.low_dim, int(self.low_dim*0.75), 3*self.low_dim),#10,10
            
            Bottleneck_judge(3*self.low_dim, int(self.low_dim*0.75), self.low_dim),#10,10
            
            Rearrange('(b s m) d -> b s m d', s = self.w, m = 6),
            

            

            
        )
        

        
        self.graph_clip = nn.Sequential( Rearrange('b s m d -> (b s) m d', m = 6),
                                        graph_transformer(words = 6, dim = self.low_dim, depth = num_depth, heads = num_head, dim_head = int(self.low_dim/num_head), mlp_dim = self.low_dim, dropout = 0.1),
                                        take_cls(keepdim = True),
                                        Rearrange('(b s) n d -> b s n d', s = self.w, d = self.low_dim, n = 1),
                                        )

        
    
        self.txt_clip = nn.Sequential(Rearrange('b n s -> (b n) s', s = 10, n = txt_size),
                            txt_mask_transformer(dict_size = 14, words = 10, dim = self.low_dim, depth = num_depth*2, 
                                                 heads = num_head, dim_head = int(self.low_dim/num_head), mlp_dim = self.low_dim, dropout = 0.1,is_pgm = True),
                            take_cls(),
                            Rearrange('(b n) d -> b n d', n = txt_size)) #b,336,d
        
        
        
        
        
        
    
   
    
    
    def forward(self, x):
 
        
        b, n, h, w = x.shape
        

        

        x = x.view(b*n, 1, h, w)
 
        x = self.vit(x)
        
        x = self.g_function(x)
        
        x = self.graph_clip(x)
        
        x_1 = x[:,:int(self.w*0.5)].mean(dim = 1)
        
        x_2 = x[:,int(self.w*0.5):].mean(dim = 1)
        
        
        y = self.txt_clip(self.txt_data.to(x.device))

        return x_1, x_2, y
        
    #"""
    
        


    
    
    def loss_function_sl(self, *out, target):


        graph = out[0]


        txt = out[1].mean(dim  = 0 ,keepdim = True)
        
        
        target_mask = target.ne(7775)

        
        graph = graph[target_mask]
        
        target = target[target_mask]
        
        
        if target.shape[0] == 0:
            
            return 0, torch.zeros(1).to(graph.device)
        
        else:


        
            r = F.cosine_similarity(graph,txt, dim = -1)
            
    
    
            loss_1 = F.cross_entropy(r / self.temperature, target)
            
    
    
    
            return loss_1, (r.argmax(dim = -1) == target).float().sum()
    
    
    
    def loss_function(self, *out, target_shape, target_line):
        

        x_shape, x_line, y = out
        
        
        loss_1, right_shape = self.loss_function_sl(x_shape, y, target = target_shape)
        
        loss_2, right_line = self.loss_function_sl(x_line, y, target = target_line)


        return loss_1 + loss_2, right_shape, right_line
    
        
        

        
def transpose(x):
    return x.transpose(-2, -1).contiguous()
    
def mul_dot(a, b):
    
    assert a.dim() == b.dim() and a.dim() == 3 and b.shape[1] == 7776  and a.shape[1] == 1, 'error'
    

    return (a@transpose(b)).squeeze(-1)
    
    
def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]          
        
        
def reasoning(*args):
	return raven_clip()

 
if __name__ == '__main__':
    x = torch.randn(10,9,80,80).cuda()
    y = torch.randint(1,(10,7776,16)).long().cuda()
    target = torch.randint(7776,(10,)).long().cuda()
    label = torch.randint(8,(10,)).long().cuda()
    
    model = raven_clip().cuda()
    
            
    model.cuda()

    out = model(x)
    
    l, choose = model.loss_function(*out, target = target, idx = label)
    
    
    accuracy = model.choose_accuracy(*out, idx = label)
    

    
    
