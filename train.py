# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 20:46:20 2021

@author: yuanbeiming
"""



import torch
import torch.nn as nn
from tqdm import tqdm

import numpy as np
from torchvision import transforms
t = transforms.Resize((80,80))

from set_args import args

from optimizer import start_opt
#%%
if args.dataset == 'PGM':
    import make_data as make_data

if args.dataset == 'Raven':
    import make_raven_data as make_data
    
if args.model == 'RS-TRAN':
    from RS_transformer_master import ptransformer as module
    
if args.model == 'RS-CNN':
    from RS_CNN_master import pcnn as module    
    
device = args.device    


train_set = make_data.Raven_Data(train = True,val = False)
train_set.shuffle_set(args.mini_batch)



train_loader , num_train = make_data.make_loader(train_set, args.batch_size)#3346,2529

num_train = len(train_set)


val_loader, num_val = make_data.raven_loader(args.batch_size, train = False, val = True)
print('number of train sample:{:.4f}\t \n number of val sample:{:.4f}\t'.format(num_train,  num_val))


test_loader, num_test = make_data.raven_loader(args.batch_size, train = False, val = False)



model = module.reasoning(args)

name = model.name + '_on_' + args.dataset + make_data.aaa


#%%
if torch.cuda.device_count() > 1:
  model = nn.DataParallel(model)
  print( torch.cuda.device_count())

model = model.to(device)


#%%

optimiser, lr_decay= start_opt(args, model)




#%%
max_accuracy = 0
epoch = 0


#%%


while epoch < args.epoch:
    accuracy = [0]*3

    
    accuracy_val = [0]*3
    
    
    loss_train = 0
    # loss_test_all = 0
    with tqdm(total=len(train_loader)  + len(val_loader)) as pbar:
        model.train()  #启用 Batch Normalization 和 Dropout。
        for x_train, y_train in train_loader:

            x_train = t(x_train).float().to(device)
            y_train = (y_train).long().to(device)
            
            
            x_train = x_train/255.
            out_train = model(x_train.contiguous()) #输出

            

            loss = (model.module.loss_function(*out_train, target = y_train, temperature = args.temper) if isinstance(model, nn.DataParallel) 
                    else model.loss_function(*out_train, target = y_train, temperature = args.temper))
            model.zero_grad()
            loss.backward()#回传
            optimiser.step()  
            
            with torch.no_grad():

                accuracy[0] += ((model.module.choose(*out_train).argmax(dim = -1) == y_train).float().sum()).cpu().numpy()

            loss_train += loss.detach().cpu()
            pbar.set_postfix(loss_batch = loss.detach())#进度条
            pbar.update(1)
        lr_decay.step()
        
        accuracy[0] /= num_train

    
        loss_train /= len(train_loader)
        
        train_set.shuffle_set()

        num_train = len(train_set)
        
        
        model.eval()
        with torch.no_grad():
            
            for index, (x_test, y_test) in enumerate(val_loader):
                x_test = t(x_test).float().to(device)
                y_test = y_test.long().to(device)
                
                x_test = x_test/255.
                
                out_test = model(x_test.contiguous())


                
                accuracy_val[0] += ((model.module.choose(*out_test).argmax(dim = -1) == y_test).float().sum()).cpu().numpy()

               
                pbar.update(1)

                
            accuracy_val[0] /= num_val
            
            
            

    
    # Stores model
    if accuracy_val[0] > max_accuracy: 
        torch.save(model.module.state_dict()  if isinstance(model, nn.DataParallel) else model.state_dict(), args.save_path + 'model_'+ name+'_best.pt')
        torch.save(optimiser.state_dict(), args.save_path + 'optimiser_' + name+'_best.pt')

        max_accuracy = accuracy_val[0]
        

    torch.save(model.module.state_dict()  if isinstance(model, nn.DataParallel) else model.state_dict(), args.save_path + 'model_' + name+'_now.pt')
    torch.save(optimiser.state_dict(), args.save_path + 'optimiser_' + name+'_now.pt')

    # Print and log some results
    print("epoch:{}\t loss_train:{:.4f}\t accuracy_train:{:.4f}\t accuracy_val:{:.4f}\t learning_rate:{:.8f}".format(epoch, loss_train,  accuracy[0], accuracy_val[0],  optimiser.state_dict()['param_groups'][0]['lr']))

    with open(args.log_save_path + "test_on_"+ name +".txt", "a") as f:  # 打开文件
        
        f.write("epoch:{}\t loss_train:{:.4f}\t accuracy_train:{:.4f}\t accuracy_val:{:.4f}\t  learning_rate:{:.8f}\n".format(epoch, loss_train,  accuracy[0], accuracy_val[0],  optimiser.state_dict()['param_groups'][0]['lr']))
    epoch += 1
    
    
    #%%

accuracy_test = [0]*3
with tqdm(total=len(test_loader)) as pbar:

    model.eval()#不启用 Batch Normalization 和 Dropout。
    with torch.no_grad():
        
        for index, (x_test, y_test) in enumerate(test_loader):
            x_test = t(x_test).float().to(device)
            y_test = y_test.long().to(device)
            
            x_test = x_test/255.
            
            out_test = model(x_test.contiguous())


            
            accuracy_test[0] += ((model.module.choose(*out_test).argmax(dim = -1) == y_test).float().sum()).cpu().numpy()
            pbar.update(1)

            
        accuracy_test[0] /= num_test
        
    print("accuracy_test:{:.4f}\t ".format(accuracy_test[0]))
    with open(args.log_save_path + "test_on_"+ name +".txt", "a") as f:

        f.write("accuracy_test:{:.4f}\t ".format(accuracy_test[0])+ '\n')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
