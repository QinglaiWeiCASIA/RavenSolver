#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:28:51 2022

@author: yuanbeiming
"""
import argparse
import torch

def str2bool(val):
  if val == "True":
    return True
  if val == "False":
    return False

parser = argparse.ArgumentParser()

# General Arguments
parser.add_argument("-dataset", help="Raven or PGM",
                    type=str, default='PGM')
parser.add_argument("-datapath", help="path list of Raven or PGM",
                    type=list or tuple, default=['PGM'])

parser.add_argument("-mini_batch", help="size of mini batch",
                    type=int, default=200000)

parser.add_argument("-device", help="cuda or cpu",
                    type=str, default= torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

parser.add_argument("-model", help="RS-CNN or RS-TRAN",
                    type=str, default= 'RS-TRAN')

parser.add_argument("-lr", help="Learning Rate",
                    type=float, default=0.001)

parser.add_argument("-sch_step", help="lr_scheduler_step",
                    type=int, default=1)

parser.add_argument("-sch_gamma", help="lr_scheduler_gamma",
                    type=float, default=0.99)

parser.add_argument("-batch_size", help="Batch Size",
                    type=int, default=500)
parser.add_argument("-epoch", help="Epochs to Train",
                    type=int, default=400)
parser.add_argument("-opt", help="Choose optimizer",
                    type=str, default='Adam')

parser.add_argument("-weight_decay", help="Weight Decay value",
                    type=float, default=1e-6)


# Network Arguments
parser.add_argument("-big", help="Model size",
                    type=str2bool, default='False')

parser.add_argument("-more_big", help="large Model size",
                    type=str2bool, default='False')

parser.add_argument("-dropout", help="dropout",
                    type=str2bool, default='False')

parser.add_argument("-temper", help="temperature for RS-CNN",
                    type=str2bool, default='0.2')

parser.add_argument("-save_path", help="model and optimizer save path",
                    type=str, default='./')

parser.add_argument("-log_save_path", help="log save path",
                    type=str, default='./')





# Dataset Arguments

args = parser.parse_args()

