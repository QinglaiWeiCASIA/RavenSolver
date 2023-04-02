This code is for paper: 
Qinglai Wei, Diancheng Chen, Beiming Yuan, Multi-Viewpoint and Multi-Evaluation with Felicitous Inductive Bias Boost Machine Abstract Reasoning Ability. Preprint at https://arxiv.org/abs/2210.14914


Datasets evaluated in this work:
1. RAVEN: http://wellyzhang.github.io/project/raven.html
2. I-RAVEN: https://github.com/husheng12345/SRAN
3. PGM: https://github.com/deepmind/abstract-reasoning-matrices

As mentioned in the paper, the network architecture of perception module in RS-CNN is the same as the multi-scale encoder in MRNet: https://github.com/yanivbenny/MRNet 

======================================================================


RS-CNN and RS-TRAN are models for solving RPM problems.

RS-TRAN-CLIP is a model for predicting the meta-data, while the perception module of the trained RS-TRAN-CLIP can be used as a pretraining model in RS-TRAN. 

======================================================================


Experiment parameters:

RS-CNN on RAVEN & I-RAVEN: 

center: lr: 1e-3, lr_decay: 0.99, batch size: 500, weight decay: 0, optimizer: Adam, InfoNCE temperature: 0.2.

distribute four: lr: 1e-3, lr_decay: 0.99, batch size: 500, weight decay: 0, optimizer: Adam, InfoNCE temperature: 0.2.

distribute nine: lr: 1e-3, lr_decay: 0.99, batch size: 500, weight decay: 0, optimizer: Adam, InfoNCE temperature: 0.01 for RAVEN, 0.05 for I-RAVEN.

up down: lr:1e-3, lr_decay: 0.99, batch size: 500, weight decay: 0, optimizer: Adam, InfoNCE temperature: 0.2.

left right: lr:1e-3, lr_decay: 0.99, batch size: 500, weight decay: 0, optimizer: Adam, InfoNCE temperature: 0.2.

OIC: lr: 1e-3, lr_decay: 0.99, batch size: 500, weight decay: 0, optimizer: Adam, InfoNCE temperature: 0.2.

OIG: lr: 1e-3, lr_decay: 0.99,  batch size: 500, weight decay: 1e-6, optimizer: Adam, InfoNCE temperature: 0.2.


======================================================================

MoM training paradigm: PGM is a very large dataset, for the convenience of observing and hype-parameter fine-tuning, for each epoch, we only train a minibatch of the training set (randomly draw from the training set). Given that each minibatch will be divided into smaller minibatches in each epoch, we call the training paradigm as minibatch of minibatch (MoM). MoM size means the training set size for each epoch. 


======================================================================


RS-CNN on PGM:
Netrual: lr: 1e-3, lr_decay: 0.99, batch size: 500, weight decay: 0, optimizer: Adam, temperature: 0.05, MoM size: 200000.

======================================================================


RS-TRAN on Raven & I-Raven :

center: lr:1e-3, lr_decay: 0.99, batch size: 500, weight decay: 0, optimizer: Adam.

distribute four: lr: 1e-3, lr_decay: 0.99, batch size: 500, weight decay: 0, optimizer: Adam.

distribute nine: lr: 1e-3, lr_decay: 0.99, batch size: 500, weight decay: 0, optimizer: Adam.

up down: lr: 1e-3, lr_decay: 0.99, batch size: 500, weight decay: 0, optimizer: Adam.  

left right: lr:1e-3, lr_decay: 0.99, batch size: 500, weight decay: 0, optimizer: Adam.

OIC: lr:1e-3, lr_decay: 0.99, batch size: 500, weight decay: 0, optimizer: Adam.

OIG: lr: 1e-3, lr_decay: 0.99, batch size: 500, weight decay: 0, optimizer: Adam.
 

======================================================================



RS-TRAN on PGM:

Netrual: lr: 1e-3, lr_decay: 0.99, batch size: 500, weight decay: 0, optimizer: Adam, MoM size: 200000.
Others: lr: 1e-3, lr_decay: 0.99, batch size: 500, weight decay: 0, optimizer: Adam, MoM: 200000.


======================================================================


Arguments in the code: 
set all the arguments in set_args.py:

Namespace(

batch_size=500,

big=False, //True when training RS-CNN on O-IG.

datapath,

dataset, //'RAVEN' or 'PGM'

device, //0 when default

dropout=False,

epoch, //maximum epoch number

log_save_path='./', //path for log data

lr=0.001, //learning rate

mini_batch=200000, //for MoM training paradigm

model, //'RS-TRAN' or 'RS-CNN'

more_big=False //model with more channels

opt='Adam', //optimizer

save_path='./', //path for saving model parameters

sch_gamma=0.99, sch_step=1, //parameter for lr_scheduler.StepLR

weight_decay=1e-06 
)
