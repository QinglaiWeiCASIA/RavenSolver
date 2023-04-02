How to use these codes

Save RAVEN dataset to 'Clip/Datasets/', e.g. Clip/Datasets/distribute_nine/

Save PGM dataset to 'Clip/'，e.g. Clip/neutral/


Run

	"Clip/read_tokens_pgm.py",
	"Clip/Datasets/read_tokens_raven_d9.py",
	"Clip/Datasets/read_tokens_raven_oig.py"  
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

	"Clip/train_tran_clip_d9.py",
	"Clip/train_tran_clip_oig.py",
	"Clip/train_tran_clip_pgm.py"
	

========================================================
