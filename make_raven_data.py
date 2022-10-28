#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:03:10 2022

@author: yuanbeiming
"""
import numpy as np
import os
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader 
import random
from tqdm import tqdm
from set_args import args

#in_distribute_four_out_center_single  distribute_nine

if isinstance(args.datapath, list) or isinstance(args.datapath, tuple):
    path = args.datapath

if isinstance(args.datapath, str):
    path = [args.datapath]

file_names = [None]*len(path)

aaa = path[0].split('/')[-2]
num = 0
for i in range(len(path)):
    

    file_names[i] = os.listdir(path[i])
    num += len(file_names[i])
    # random.shuffle(file_names[i])
    

train_file =  []
test_file = []
val_file = []

with tqdm(total=num) as pbar:
    for i, file_name in enumerate(file_names):#一个path下所有文件名
        for file in file_name:

            if file.split('_')[2] == 'test.npz':
                test_file.append(path[i]+file)
                
                
            if file.split('_')[2] == 'val.npz':
                val_file.append(path[i]+file)
                
                
            if file.split('_')[2] == 'train.npz':
                train_file.append(path[i]+file)
            pbar.update(1)
        
random.shuffle(train_file)

    
random.shuffle(test_file)


random.shuffle(val_file)



class Raven_Data(Data.Dataset):
    def __init__(self,
                 train = True,
                 
                 val = False,
                 train_file = train_file,
                 test_file = test_file,
                 val_file = val_file):
        super(Raven_Data, self).__init__()
        # print(domains)

    

        if train == True and val == False:
            self.file_names_npz = train_file[:]
            print('training')
                


       
        if train == False and val == False:
            self.file_names_npz = test_file[:]
            print('testing')


            
            
        if train == False and val == True:
            self.file_names_npz = val_file[:]
            print('valing') 

        assert len(self.file_names_npz) != 0 , 'empty error' 
        

        self.work_npz = self.file_names_npz
        
        
        
    def shuffle_set(self, num = 200000):
        random.shuffle(self.file_names_npz)
        
        self.work_npz = self.file_names_npz[:num]
                
    def __getitem__(self, index):

        

    
            target = np.load(self.work_npz[index])
    

    
            return target['image'].reshape(16,160,160),  target['target']

    
    
    
    def __len__(self):
        return len(self.work_npz)

def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    bboxes = np.array(bboxes)
    return torch.Tensor(images), torch.Tensor(bboxes)

def raven_loader(batch_size,  train = True, val = False, num_workers = 8):
    dataset = Raven_Data(train = train, val = val)
    return DataLoader(dataset, batch_size = batch_size, sampler = None, collate_fn = yolo_dataset_collate, shuffle = True, num_workers = num_workers), len(dataset)



def make_loader(dataset, batch_size,  num_workers = 8):

    return DataLoader(dataset, batch_size = batch_size, sampler = None, collate_fn = yolo_dataset_collate, shuffle = True, num_workers = num_workers), len(dataset)