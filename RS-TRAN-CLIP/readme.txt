How to use these codes

Save RAVEN dataset to 'RS-TRAN-CLIP/Datasets/', e.g. RS-TRAN-CLIP/Datasets/distribute_nine/

Save PGM dataset to 'RS-TRAN-CLIP/'，e.g. Clip/neutral/


Run

	"RS-TRAN-CLIP/read_tokens_pgm.py",
	"RS-TRAN-CLIP/Datasets/read_tokens_raven_d9.py",
	"RS-TRAN-CLIP/Datasets/read_tokens_raven_oig.py"  
to obtain meta-data in pkl form.	

=======================================================

Set configurations in 'set_args':：
Namespace(
batch_size=500, 
big=False, 

device, 
dropout=False, 
epoch=400, 

lr=0.001, 
mini_batch=200000, #Minibatch of minibatch training

sch_gamma=0.99, 
sch_step=1, 
weight_decay=0 
)


========================================================
run

	"RS-TRAN-CLIP/train_tran_clip_d9.py",
	"RS-TRAN-CLIP/train_tran_clip_oig.py",
	"RS-TRAN-CLIP/train_tran_clip_pgm.py"
	

========================================================
To improve the performance of RS-TRAN in terms of convergence speed and reasoning accuracy, use the perception module of the trained RS-TRAN-CLIP in RS-TRAN as pretraining model. 

