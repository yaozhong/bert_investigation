import torch
import optuna

from data_loading import *
from models import *

import multiprocessing as mp
import torch.optim as optim

from tbfs_eval import evaluate

# define the global accessible file
kmer=5
epoch=15
data_path="/ws1/pretrain/randpt/data/TBFS/motif_occupancy/wgEncodeAwgTfbsBroadDnd41CtcfUniPk/"
embedding="dnabert"
embedding_file="/ws1/pretrain/DNABERT/ptModel/5-new-12w-0"
device="cuda"


def set_seed(s):                                                                                              
	random.seed(s)
	np.random.seed(s)
	torch.manual_seed(s)

	torch.cuda.manual_seed_all(s)
	#add additional seed
	torch.backends.cudnn.deterministic=True
	torch.use_deterministic_algorithms = True


# evaluating function
def objective(trial: optuna.trial.Trial) -> float:

	# kernel length
	kernel1 = trial.suggest_int("kernel_1", 5, 50, step=6)
	kernel2 = trial.suggest_int("kernel_2", 5, 50, step=6)
	kernel3 = trial.suggest_int("kernel_3", 5, 50, step=6)
	kernel_size = [kernel1, kernel2, kernel3]

	dropout = trial.suggest_float("dropout_rate", 0.1, 0.4, step=0.1)
	#lstm_hidden = trial.suggest_int("lstm_hidden", 32, 128, step=32)
	fc_hidden = trial.suggest_int("fc_hidden", 32, 256, step=32)

	batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
	optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
	lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

	# data loadding
	cores = mp.cpu_count()

	# data generator
	train_file_name = data_path + "/train.data"
	
	# generate different embeddings
	if embedding == "onehot":
		dic = get_1hot_embedding_dic(kmer)
		embed_dim = len(dic.keys())
	elif embedding == "dna2vec":
		dic = get_dna2vec_embedding_dic(embedding_file, kmer)
		embed_dim = 100
	elif embedding == "dnabert":
		dic = get_bert_embedding(embedding_file)
		embed_dim = 768
	else:
		print("Please check provided embedding files ...")
		exit(-1)

	vocab = dic.keys()
	train_dataset = TBFS_dataset(train_file_name, kmer, dic)
	
	# split the dataset
	train_labels = train_dataset.get_label_list()
	train_idx, dev_idx= train_test_split(np.arange(len(train_labels)), test_size=0.2, shuffle=True, stratify=train_labels)
	train_ds_split = Subset(train_dataset, train_idx)
	dev_ds_split = Subset(train_dataset, dev_idx)

	train_ds_generator = DataLoader(train_ds_split, batch_size, collate_fn=partial(my_collate_fn, kmer=kmer, dic=dic), num_workers=cores, worker_init_fn=fix_worker_init_fn) 
	dev_ds_generator = DataLoader(dev_ds_split, batch_size, collate_fn=partial(my_collate_fn, kmer=kmer, dic=dic), num_workers=cores, worker_init_fn=fix_worker_init_fn) 
	

	## define the model space and trails
	# define space
	net = CNN_biLSTM_ht(embed_dim, kernel_size, kmer, dropout, fc_hidden)
	net.to(device)

	criterion = nn.CrossEntropyLoss()

	# generate the optimiers.
	optimizer = getattr(optim, optimizer_name)(net.parameters(), lr)

	# Training of the model
	best_mcc = -5
	for ep in range(epoch):
		#epoch_loss = 0
		for X, Y, category in train_ds_generator:

			X = torch.tensor(X, dtype = torch.float32).transpose(1,2)
			
			optimizer.zero_grad()
			pred = net(X.to(device))
			loss = criterion(pred, torch.tensor(Y).to(device).long())
			loss.backward()
			optimizer.step()

			#train_loss += loss.item()
			#epoch_loss += loss.item()

		mcc, auc_score, auprc = evaluate(net, dev_ds_generator, device, "DEV",  False)

		if mcc > best_mcc:
			best_mcc = mcc

	#trail.report(best_mcc, epoch)

	#if trial.should_prune():
	#	raise optuna.exceptions.TrialPruned()

	return best_mcc

if __name__ == "__main__":

	set_seed(123)

	study = optuna.create_study(direction="maximize")
	study.optimize(objective, n_trials=50)

	pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrailState.PRUNED]
	complete_trials = [t for t in study.trials if t.state == optuna.trial.TrailState.COMPLETE]

	print("== Trails information ==")
	print("* Finished [%d] trails, pruned [%d] trails, completed [%d] trails" %(len(study.trials), len(pruned_trails), len(complete_trials)))
	trial = study.best_trial
	print("@ Best Trail:")
	print("- Value:", trial.value)
	print("- Params:")
	for key, value in trial.params.items():
		print("	{}:{}".format(key, value))



