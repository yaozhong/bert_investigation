# training model

import argparse, time
from data_loading import *
from models import *

import multiprocessing as mp
import torch.optim as optim

from tata_eval import evaluate, evaluate_cache, evaluate_withPosition


# random seed fix
def set_seed(s):                                                                                              
	random.seed(s)
	np.random.seed(s)
	torch.manual_seed(s)

	torch.cuda.manual_seed_all(s)
	#add additional seed
	torch.backends.cudnn.deterministic=True
	torch.use_deterministic_algorithms = True


def train(model, data_path, model_path, kmer, kernel_size = [27, 14, 7], \
	embedding="onehot", embedding_file="", device="cpu", \
	batch_size=128, lr=0.001, epoch=20, dropout=0.1, use_position=False, num_worker=72):

	train_start = time.time()

	# set up  devices
	cores = mp.cpu_count()
	if num_worker > 0 and num_worker < cores:
		cores = num_worker

	# data generator
	print(" |- Start preparing train/dev dataset.")

	train_file_name = data_path + "/train.tsv"
	dev_file_name = data_path + "/dev.tsv"

	if embedding == "onehot":
		dic = get_1hot_embedding_dic(kmer)
		embed_dim = len(dic.keys())
	elif embedding == "dna2vec":
		dic = get_dna2vec_embedding_dic(embedding_file, kmer)
		embed_dim = 100
	elif embedding == "dnabert":
		dic, p_vecs = get_bert_embedding(embedding_file)
		embed_dim = 768
	else:
		print("Please check provided embedding files ...")
		exit(-1)

	vocab = dic.keys()

	train_dataset = TATA_dataset(train_file_name, kmer, dic)
	dev_dataset = TATA_dataset(dev_file_name, kmer, dic)

	train_ds_generator = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=partial(my_collate_fn, kmer=kmer, dic=dic), num_workers=cores, worker_init_fn=fix_worker_init_fn) 
	dev_ds_generator = DataLoader(dev_dataset, batch_size, collate_fn=partial(my_collate_fn, kmer=kmer, dic=dic), num_workers=cores, worker_init_fn=fix_worker_init_fn)

	# do the data caching for the speed up

	net = globals()[model](embed_dim, kernel_size, kmer, dropout)
	net.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=lr)

	print(" |- Total [%s] Parameters %d:" %(model, sum([p.nelement() for p in net.parameters()])))
	
	print(" |- Training started ...")

	train_loss, epoch_loss, best_mcc = 0, 0, -5

	model_tag = ""
	if embedding_file != "":
		model_tag = embedding_file.split("/")[-1]
	model_save_name = model_path + "/"+ model + "-" +embedding  \
				+ "-b" + str(batch_size) + "-lr"+ str(lr) + \
				"-kernel_" + "_".join([ str(k) for k in kernel_size ]) + \
				"-dropout" + str(dropout) + "-tag_" + model_tag + ".model" 

	# saving model
	## caching the training data during the training process
	cache_train_X, cache_train_Y, cache_train_category = [], [], []
	for X, Y, category in train_ds_generator:

		if use_position:
			p_mat = np.tile(p_vecs[:X.shape[1], :], (X.shape[0], 1, 1))
		else:
			p_mat = np.zeros(X.shape)

		cache_train_X.append(X+p_mat)
		cache_train_Y.append(Y)
		cache_train_category.append(category)

	cache_dev_X, cache_dev_Y, cache_dev_category = [], [], []
	for X, Y, category in dev_ds_generator:
		#X = torch.tensor(X, dtype = torch.float32).transpose(1,2)

		if use_position:
			p_mat = np.tile(p_vecs[:X.shape[1], :], (X.shape[0], 1, 1))
		else:
			p_mat = np.zeros(X.shape)

		cache_dev_X.append(X+p_mat)
		cache_dev_Y.append(Y)
		cache_dev_category.append(category)

	## train loop
	for ep in range(epoch):
		epoch_loss = 0
		for i in range(len(cache_train_X)):
			X, Y, category = cache_train_X[i], cache_train_Y[i], cache_train_category[i]
		#for X, Y, category in train_ds_generator:

			X = torch.tensor(X, dtype = torch.float32).transpose(1,2)
			
			optimizer.zero_grad()
			pred = net(X.to(device))
			loss = criterion(pred, torch.tensor(Y).to(device).long())
			loss.backward()
			optimizer.step()

			train_loss += loss.item()
			epoch_loss += loss.item()

		print("Epoch-%d, Loss=%f" %(ep,epoch_loss))
		#mcc, auc_score, auprc = evaluate(net, dev_ds_generator, device)
		mcc, auc_score, auprc, f1 = evaluate_cache(net, [cache_dev_X, cache_dev_Y, cache_dev_category], device,"DEV", True)

		if mcc > best_mcc:
			best_mcc = mcc
			# save the model 
			torch.save(net.state_dict(), model_save_name)

	return model_save_name

# using the best model on the testset
def test(model, data_path, model_path, kmer, kernel_size = [27, 14, 7], \
	     embedding="onehot", embedding_file="", device="cpu", dropout=0.1, use_position=False):

	test_file_name = data_path + "/test.tsv"

	# set up  devices
	cores = mp.cpu_count()

	if embedding == "onehot":
		dic = get_1hot_embedding_dic(kmer)
		embed_dim = len(dic.keys())
	elif embedding == "dna2vec":
		dic = get_dna2vec_embedding_dic(embedding_file, kmer)
		embed_dim = 100
	elif embedding == "dnabert":
		dic, p_vecs = get_bert_embedding(embedding_file)
		embed_dim = 768
	else:
		print("Please check provided embedding files ...")
		exit(-1)

	vocab = dic.keys()

	test_dataset = TATA_dataset(test_file_name, kmer, dic)
	test_ds_generator = DataLoader(test_dataset, 256, collate_fn=partial(my_collate_fn, kmer=kmer, dic=dic),\
	 num_workers=cores, worker_init_fn=fix_worker_init_fn)

	net = globals()[model](embed_dim, kernel_size, kmer, dropout)
	net.load_state_dict(torch.load(model_path))
	net.to(device)

	if use_position == False:
		evaluate(net, test_ds_generator, device, "TEST")
	else:
		evaluate_withPosition(net, test_ds_generator, device, "TEST", p_vecs)


if __name__ == "__main__":

	set_seed(123)

	parser = argparse.ArgumentParser(description='<TATA Promoter Prediction Task>')

	parser.add_argument('--kmer',default=1,      type=int,  required=True, help="Kmer size for the input.")
	parser.add_argument('--model',     default='deepromoter',   type=str, required=True,  help="DL models [CNN_biLSTM/deepPromoterNet].")
	parser.add_argument('--model_dir', action="store",   type=str, required=True,  help="directory for saving the trained model.")
	parser.add_argument('--device',       default="cpu", type=str, required=False, help='GPU Device(s) used for training')
	
	parser.add_argument('--data_dir', action="store",   type=str, required=True,  help="directory of data.")
	
	parser.add_argument('--lr',     	default=0.001,   type=float, required=False, help='Learning rate')
	parser.add_argument('--epoch',      default=20,       type=int, required=False, help='Training epcohs')
	parser.add_argument('--batch_size' ,default=64,      type=int,  required=False, help="batch_size of the training.")
	parser.add_argument('--dropout'    ,default=0.1,      type=float,  required=False, help="Dropout rate.")

	parser.add_argument('--embedding', default="onehot",  type=str,  required=True, help="The embedding type")
	parser.add_argument('--embedding_file', default="",  type=str,  required=False, help="The embedding file")

	parser.add_argument('--position', action='store_true')
	parser.add_argument('--cnn_kernel_size', default="24,17,7", type=str, required=False, help="Kernel size used in CNN.")

	args = parser.parse_args()

	cnn_kernel_size = [ int(k) for k in args.cnn_kernel_size.split(',')]

	# model train
	model_save_name = train(args.model, args.data_dir, args.model_dir, args.kmer, cnn_kernel_size, \
		args.embedding, args.embedding_file, \
		args.device, args.batch_size, args.lr, args.epoch, args.dropout, args.position)

	# model evaluation
	print("\n @ ---- Testset evaluation ----")
	test(args.model, args.data_dir, model_save_name, args.kmer, cnn_kernel_size, \
		args.embedding, args.embedding_file, args.device, args.dropout, args.position)

