#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:03:10 2022

@author: yuanbeiming
"""
import numpy as np
import pickle as pkl
import os
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
import random
from tqdm import tqdm


path = ['./neutral/']
file_names = [None]*len(path)

aaa = path[0].split('/')[-2]
print(aaa)
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

            if file.split('_')[2] == 'test':
                test_file.append(path[i]+file)
                
            if file.split('_')[2] == 'val':
                val_file.append(path[i]+file)
                
                
            if file.split('_')[2] == 'train':
                train_file.append(path[i]+file)
            pbar.update(1)
        
random.shuffle(train_file)
print(len(train_file))
    
random.shuffle(test_file)
print(len(test_file))

random.shuffle(val_file)
print(len(val_file))

path = './Pgm_neutral_tokens.pkl'
f=open(path,'rb')

data = pkl.load(f)

# txt_data_dict = data['tokens']

label_dict_shape = data['label_shape']
label_dict_line = data['label_line']
class Raven_Data(Data.Dataset):
    def __init__(self,
                 train = True,
                 
                 val = False,
                 train_file = train_file,
                 test_file = test_file,
                 val_file = val_file):
        super(Raven_Data, self).__init__()
        # print(domains)

    

        if train == True:  
            self.file_names_npz = train_file[:]
            print('train')
                


       
        if train == False and val == False:
            self.file_names_npz = test_file[:]
            print('test')
            
        if train == False and val == True:
            self.file_names_npz = val_file[:]
            print('val')

        assert len(self.file_names_npz) != 0 , 'error' 
        

        self.work_npz = self.file_names_npz
        
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
        
        self.txt_data = []
        for c in range(8,14):#color
            for n in range(8,14):#number
                for p in range(8,14):#position
                    for s in range(8,14):#size
                        for t in range(8,14):#type
                            self.txt_data.append(np.array([ 3,  c,  4, n,  5, p,  6, s, 7, t]))
                            
                            
                            
        self.txt_data = np.array(self.txt_data)
        
        
    def shuffle_set(self, num = 200000):
        random.shuffle(self.file_names_npz)
        
        self.work_npz = self.file_names_npz[:num]
                
    def __getitem__(self, index):

        

    
            opt_path = self.work_npz[index]
            target = np.load(opt_path)

            idx = target['target']
            
            a = int(label_dict_shape[opt_path.split('/')[-1]])
            b = int(label_dict_line[opt_path.split('/')[-1]])

    
            return target['image'].reshape(16,160,160)[[0,1,2,3,4,5,6,7,idx+8],], a, b, idx

    
    
    
    def __len__(self):
        return len(self.work_npz)

def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    A =[]
    B = []
    for img, a_out, b_in, idx in batch:
        images.append(img)
        bboxes.append(idx)
        A.append(a_out)
        B.append(b_in)
    images = np.array(images)
    bboxes = np.array(bboxes)
    A=np.array(A)
    
    B=np.array(B)
    return torch.Tensor(images), torch.Tensor(A), torch.Tensor(B), torch.Tensor(bboxes),

def raven_loader(batch_size,  train = True, val = False, num_workers = 16):
    dataset = Raven_Data(train = train, val = val)
    return DataLoader(dataset, batch_size = batch_size, sampler = None, collate_fn = yolo_dataset_collate, shuffle = True, num_workers = num_workers, pin_memory=True, prefetch_factor=2), len(dataset)



def make_loader(dataset, batch_size,  num_workers = 16):

    return DataLoader(dataset, batch_size = batch_size, sampler = None, collate_fn = yolo_dataset_collate, shuffle = True, num_workers = num_workers, pin_memory=True, prefetch_factor=2), len(dataset)