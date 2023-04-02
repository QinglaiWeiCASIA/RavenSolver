# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 20:46:20 2021

@author: yuanbeiming
"""



import torch
import torch.nn as nn
from tqdm import tqdm


from Tran_clip import tran_clip_pgm as model_vit





import numpy as np
from torchvision import transforms



t = transforms.Resize((80,80))

import make_pgm_clip_data as make_data
from set_args import args

batch_size = args.batch_size


weight_decay = args.weight_decay


train_set = make_data.Raven_Data(train = True,val = False)
len_train_set = len(train_set)




train_loader , num_train = make_data.make_loader(train_set, batch_size)#3346,2529
print('number before shuffle:', num_train)

print(num_train, len(train_loader))

train_set.shuffle_set(args.mini_batch)

num_train = len(train_set)


print('number after shuffle:', num_train)


print(num_train, len(train_loader))

val_loader, num_val = make_data.raven_loader(batch_size, train = False, val = True)


print('train:', num_train, 'test:', num_val)

device = args.device

model = model_vit.raven_clip()

if model_vit.big == True:
	name =  'big_' + model.name +  '_' +str(len_train_set) + '_' + make_data.aaa
	text_name =  'big_' + model.name  + make_data.aaa
else:
	name =  model.name +  '_' +str(len_train_set) + '_' + make_data.aaa
	text_name =  model.name +  make_data.aaa
print(name)


	

#%%
if torch.cuda.device_count() > 1:



  model = nn.DataParallel(model)
  print( torch.cuda.device_count())

model = model.to(device)


#%%

optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = weight_decay)


print('lr:', optimiser.state_dict()['param_groups'][0]['lr'])
print('w_d:', optimiser.state_dict()['param_groups'][0]['weight_decay'])

lr_decay = torch.optim.lr_scheduler.StepLR(optimiser, step_size= args.sch_step, gamma= args.sch_gamma)



#%%
max_accuracy = 0
num_epoch = args.epoch
epoch = 0


#%%
with open("test_on_"+text_name+".txt", "a") as f:  # 打开文件
        f.write('train_num_sample:' + str(len_train_set)+ '\n')
        f.write('temperature:' + str(model_vit.temperature)+ '\n')
        f.write('weight_decay:' + str(weight_decay)+ '\n')

print('temperature:',model_vit.temperature)
print('weight_decay:' ,(weight_decay))

while epoch < num_epoch:
    accuracy = [0]*5
    
    drop_sample = [0]*2
    
    drop_test_sample = [0]*2

    
    accuracy_val = [0]*5
    
    loss_train = 0
    # loss_test_all = 0
    with tqdm(total=len(train_loader)  + len(val_loader)) as pbar:
        model.train()  #启用 Batch Normalization 和 Dropout。
        for x_train, label,  label_in, idx in train_loader:


            # Model training
           
            
            #梯度清零

            x_train = t(x_train).float().to(device)


            idx = (idx).long().to(device)

            label = label.long().to(device)

            label_in = label_in.long().to(device)
            
            
            
            drop_sample[0] += label.eq(7775).sum().item()
            
            drop_sample[1] += label_in.eq(7775).sum().item()
            
            
            
            
            
            x_train = x_train/255.

            out_train = model(x_train) #输出

            

            loss, right, right_in = (model.module.loss_function(*out_train, target_shape = label, target_line = label_in, idx = idx) if isinstance(model, nn.DataParallel) 
                    else model.loss_function(*out_train, target_shape = label, target_line = label_in, idx = idx))

            # choose_right, choose_right_in, choose_right_all = model.module.choose_accuracy(*out_train, idx = idx)
            model.zero_grad()
            loss.backward()#回传
            optimiser.step()  
            
            with torch.no_grad():

                accuracy[0] += right.cpu().numpy()
                accuracy[1] += right_in.cpu().numpy()
               
                



            loss_train += loss.item()
            pbar.set_postfix(loss_batch = loss.item())#进度条
            pbar.update(1)
        lr_decay.step()
        
        if num_train - drop_sample[0] == 0:
            accuracy[0] = 0
        else:
            accuracy[0] /= (num_train - drop_sample[0])
            
            
        if num_train - drop_sample[1] == 0:
            accuracy[1] = 0
        else:
            accuracy[1] /= (num_train - drop_sample[1])
       

    
        loss_train /= len(train_loader)
        train_set.shuffle_set(args.mini_batch)
        num_train = len(train_set)


        # num_train = len(train_set)
        
        # 测试错误率和损失函数
        
        model.eval()#不启用 Batch Normalization 和 Dropout。
        with torch.no_grad():
            
            for index, (x_test, label,  label_in, idx)in enumerate(val_loader):
 
                x_test = t(x_test).float().to(device)

                label = label.long().to(device)
                label_in = label_in.long().to(device)
                
                
                drop_test_sample[0] += label.eq(7775).sum().item()
                
                drop_test_sample[1] += label_in.eq(7775).sum().item()
                
                
                
                idx = (idx).long().to(device)
                x_test = x_test/255.
                
                out_test = model(x_test)

                _, right, right_in = (model.module.loss_function(*out_test, target_shape = label, target_line = label_in, idx = idx) if isinstance(model, nn.DataParallel) 
                        else model.loss_function(*out_test, target_shape = label, target_line = label_in, idx = idx))

                # choose_right, choose_right_in, choose_right_all = model.module.choose_accuracy(*out_test, idx = idx)
                
                accuracy_val[0] += right.cpu().numpy()
                accuracy_val[1] += right_in.cpu().numpy()
                

                pbar.update(1)

                
            if num_val - drop_test_sample[0] == 0:
                accuracy_val[0] = 0
            else:
                accuracy_val[0] /= (num_val - drop_test_sample[0])
                
                
            if num_val - drop_test_sample[1] == 0:
                accuracy_val[1] = 0
            else:
                accuracy_val[1] /= (num_val - drop_test_sample[1])
           
            
            
            

    
    # Stores model
    if accuracy_val[0] > max_accuracy: 
        torch.save(model.module.state_dict()  if isinstance(model, nn.DataParallel) else model.state_dict(), './model_'+name+'_best.pt')
        torch.save(optimiser.state_dict(), './optimiser_'+name+'_best.pt')

        max_accuracy = accuracy_val[0]
        

    torch.save(model.module.state_dict()  if isinstance(model, nn.DataParallel) else model.state_dict(), './model_'+name+'_now.pt')
    torch.save(optimiser.state_dict(), './optimiser_'+name+'_now.pt')

    # Print and log some results
    print("epoch:{}\t loss_train:{:.4f}\t accuracy_train: shape:{:.4f}\t line:{:.4f}\t accuracy_val: shape:{:.4f}\t line:{:.4f}\t  learning_rate:{:.8f}".format(epoch, loss_train,  accuracy[0], accuracy[1],
                                                                                                                                                                                         accuracy_val[0],accuracy_val[1],lr_decay.get_lr()[0]))

    with open("test_on_"+text_name+".txt", "a") as f:  # 打开文件
        
        f.write("epoch:{}\t loss_train:{:.4f}\t accuracy_train: shape:{:.4f}\t line:{:.4f}\t accuracy_val: shape:{:.4f}\t line:{:.4f}\t  learning_rate:{:.8f}".format(epoch, loss_train,  accuracy[0], accuracy[1],
                                                                                                                                                                                             accuracy_val[0],accuracy_val[1],lr_decay.get_lr()[0]))
    epoch += 1
    
    
    #%%











    
# from torchvision import transforms
# from matplotlib import pyplot as plt  
# t = transforms.ToPILImage()
# from einops.layers.torch import Rearrange
# arrenge = Rearrange('c h w -> h (c w)')
# num = 6
# plt.imshow(t(arrenge(x_train[num][:3])), cmap = 'gray')
# plt.show() 


# num = 6

# for i in range(8):

#     plt.subplot(3,3,i+1)
    
#     plt.imshow(t(x_train[num][i]), cmap = 'gray')
# plt.show() 
    


# for i in range(8):
    
#     plt.subplot(2,4,i+1)
    
#     plt.imshow(t(x_train[num][i+8]), cmap = 'gray')

    
    
# plt.show()

# print(y_train[num])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
