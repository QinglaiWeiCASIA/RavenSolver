#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:26:31 2021

@author: yuanbeiming
"""
import torch
import torch.nn as nn



import torch.nn.functional as F

from Blocks_clip import *


from einops.layers.torch import Rearrange


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
big = False
dropout = False

class raven_clip(nn.Module):
    def __init__(self, *args):
        super(raven_clip,self).__init__()

        self.name = 'Clip_raven_oig_perfect'

        size = 80
        patch = 20
        
        if big:
            num_head = 8
            num_depth = 6
            self.low_dim = 256
        else:
            num_head = 4
            num_depth = 3
            self.low_dim = 128
            
        if dropout:
            _dropout = 0.1
        else:
            _dropout = 0
         
        self.txt_data = []
        for T in range(6,9):#T年
            for S in range(6,10):#S
                for C in range(6,10):#C
                    for n in range(7,10):
                         self.txt_data.append(np.array([ 3,  T,  4, S,  5, C,  1, n, 2, 0]))
                         
        for T in range(6,9):#T
            for S in range(6,10):#S
                for C in range(6,10):#C
                    for n in range(7,10):
                         self.txt_data.append(np.array([ 3,  T,  4, S,  5, C,  1, 0, 2, n]))
                         
        for T in range(6,9):#T
            for S in range(6,10):#S
                for C in range(6,10):#C
                         self.txt_data.append(np.array([ 3,  T,  4, S,  5, C,  1, 6, 2, 6]))                 
                        
        self.txt_data = torch.from_numpy(txt_data[None,:,:]).long()            

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
                                 Rearrange('b (m n) s d -> b m n s d', n = 3, m = 3),
                                 Rearrange('b m n s d -> b s m (n d)'),
                                 )#b*16, 16, dim
        
        
        
        self.g_function = nn.Sequential(
            Rearrange('b s m d -> (b s m) d'),

            Bottleneck_judge(3*self.low_dim, int(self.low_dim*0.75), 3*self.low_dim),#10,10
            
            Bottleneck_judge(3*self.low_dim, int(self.low_dim*0.75), 3*self.low_dim),#10,10
            
            Bottleneck_judge(3*self.low_dim, int(self.low_dim*0.75), self.low_dim),#10,10
            
            Rearrange('(b s m) d -> b s m d', s = self.w, m = 3)

            
        )
        

        
        self.graph_clip = nn.Sequential( Rearrange('b s m d -> (b s) m d'),
                                        graph_transformer(words = 3, dim = self.low_dim, depth = num_depth, heads = num_head, dim_head = int(self.low_dim/num_head), mlp_dim = self.low_dim, dropout = 0.1),
                                        take_cls(),
                                        Rearrange('(b s) d -> b s d', s = self.w),
                                        #Mean(dim = 1, keepdim = True)

                                        )
 
        
        
        

        self.chop = chop = 0.5

        seq_size = 10

        dict_size = 10
    
        self.txt_clip = nn.Sequential(Rearrange('b n s -> (b n) s', s = seq_size, n = 336),
                            # nn.Embedding(16, self.low_dim),
                            txt_mask_transformer(dict_size = dict_size, words = seq_size, dim = self.low_dim, depth = num_depth*2, heads = num_head, dim_head = int(self.low_dim/num_head), mlp_dim = self.low_dim, dropout = 0.1),
                            take_cls(),
                            Rearrange('(b n) d -> b n d', n = 336)) #b,336,d
        
        
        
        
        
    
    
   
    
    
    def forward(self, x):
        b, n, h, w = x.shape

        x = x.reshape(b*n, -1, h, w)
        
        x = self.vit(x)
        
        x = self.g_function(x)
        
        x = self.graph_clip(x)
        
        
        x_1 = x[:,:int(self.w*0.5)]
        
        x_2 = x[:,int(self.w*0.5):]
        assert x_1.shape == x_2.shape
        
        
        y = self.txt_clip(self.txt_data.to(x.device))
        
        # assert x.shape[-1] == y.shape[-1]*2

        return x_1.mean(dim = 1, keepdim = True), x_2.mean(dim = 1, keepdim = True), y
    



    
    def loss_function(self, *out, target, target_in):
        
        
        
        
        graph = out[0]

        graph_in = out[1]
        
        txt = out[2].mean(dim  = 0 ,keepdim = True)
        
        
        r = F.cosine_similarity(graph,txt, dim = -1)
        
        r_in = F.cosine_similarity(graph_in,txt, dim = -1)

        loss_1 = F.cross_entropy(r / self.temperature, target)
        
        loss_2 = F.cross_entropy(r_in / self.temperature, target_in)

        return loss_1+ loss_2, (r.argmax(dim = -1) == target).float().sum(), (r_in.argmax(dim = -1) == target_in).float().sum()
        
 

 
if __name__ == '__main__':
    x = torch.randn(5,9,80,80).cuda()
    y = torch.randint(16,(5,336,8)).long().cuda()
    label = torch.randint(336,(5,)).long().cuda()
    
    model = raven_clip().cuda()

    out = model(x,y)
    
    l = model.loss_function(*out, target = label, target_in = label)
    
    choose = model.choose(*out)
    
    
