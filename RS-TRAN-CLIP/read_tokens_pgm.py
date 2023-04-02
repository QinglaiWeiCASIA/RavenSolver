#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:45:07 2022

@author: yuanbeiming
"""
from xml.dom import minidom
import numpy as np
import pickle as pkl
import os
from tqdm import tqdm
operate_path = './neutral/'


tokens_dict_shape = {}
tokens_dict_line = {}
label_dict_shape = {}
label_dict_line = {}

file_names = os.listdir(operate_path)
file_names_xml = []

xml_index = 0




keep_dir = './'



txt_data = []
for c in range(8,14):#color
    for n in range(8,14):#number
        for p in range(8,14):#position
            for s in range(8,14):#size
                for t in range(8,14):#type
                    txt_data.append(np.array([ 3,  c,  4, n,  5, p,  6, s, 7, t]))
"""       

  s l  c n p s t  p x o a c n            
 [1 0  0 0 0 1 0  1 0 0 0 0]
 [1 0  0 0 1 0 0  0 0 1 0 0]
 [1 0  0 0 0 0 1  0 0 0 0 1]
 [0 1  1 0 0 0 0  0 0 0 1 0]  


                 
  """  
for file_name in file_names:
    if os.path.splitext(file_name)[1] == '.npz':
        file_names_xml.append(file_name)
        xml_index = xml_index + 1                
txt_data = np.array(txt_data)
dict_ = { 'EOF':0,
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

with tqdm(total = xml_index) as pbar:
    for i in range(xml_index):
        work_npz = operate_path + file_names_xml[i]
        
        target = np.load(work_npz)
        state = target['relation_structure']
        
        
        shape = np.array([ 3,  13,  4, 13,  5, 13,  6, 13, 7, 13])
        
        line = np.array([ 3,  13,  4, 13,  5, 13,  6, 13, 7, 13])
        
        for item in state:
            if item[0] == b'shape':
                if item[1] == b'color':
                    shape[1] = dict_[str(item[2])[2:-1]]
                elif item[1] == b'number':
                    shape[3] = dict_[str(item[2])[2:-1]]
                elif item[1] == b'position':
                    shape[5] = dict_[str(item[2])[2:-1]]
                elif item[1] == b'size':
                    shape[7] = dict_[str(item[2])[2:-1]]
                elif item[1] == b'type':
                    shape[9] = dict_[str(item[2])[2:-1]]
                
            elif item[0] == b'line':
                
                if item[1] == b'color':
                    line[1] = dict_[str(item[2])[2:-1]]
                elif item[1] == b'number':
                    line[3] = dict_[str(item[2])[2:-1]]
                elif item[1] == b'position':
                    line[5] = dict_[str(item[2])[2:-1]]
                elif item[1] == b'size':
                    line[7] = dict_[str(item[2])[2:-1]]
                elif item[1] == b'type':
                    line[9] = dict_[str(item[2])[2:-1]]
                
            else:
                raise 
                
        label_shape = ((txt_data == shape[None,:]).sum(axis = -1) == 10)
        
        label_line = ((txt_data == line[None,:]).sum(axis = -1) == 10)
        
        assert label_shape.sum() == 1 and label_line.sum() ==1
            
        tokens_dict_shape[file_names_xml[i]] = shape
        
        tokens_dict_line[file_names_xml[i]] = line
        
        label_dict_shape[file_names_xml[i]] = label_shape.argmax(axis = -1)
        label_dict_line[file_names_xml[i]] = label_line.argmax(axis = -1)
            
        # print(label_shape.argmax(axis = -1))
        pbar.update(1)
        # print(label_line.argmax(axis = -1))
assert len(tokens_dict_shape) == xml_index and len(tokens_dict_line) == xml_index and len(label_dict_shape) == xml_index and len(label_dict_line) == xml_index
if not os.path.exists(keep_dir):
    os.mkdir(keep_dir)
with open(os.path.join(keep_dir, 'Pgm_' + operate_path.split('/')[-2]) + '_tokens.pkl' ,'wb') as f:#change
    pkl.dump({'tokens_shape': tokens_dict_shape, 'tokens_line': tokens_dict_line, 'label_shape':label_dict_shape,'label_line':label_dict_line, 'txt_data':txt_data}, f, pkl.HIGHEST_PROTOCOL)
# sentences = 'in ' + sentences + ' out ' + sentences_in


#%%
