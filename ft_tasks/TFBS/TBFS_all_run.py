# systematic running the results on the TFBS
import os.path
import argparse, time
from data_loading import *
from models import *
from tbfs_train import *

import multiprocessing as mp
import torch.optim as optim

from tbfs_eval import evaluate
import pandas as pd


def set_seed(s):                                                                                              
	random.seed(s)
	np.random.seed(s)
	torch.manual_seed(s)

	torch.cuda.manual_seed_all(s)
	#add additional seed
	torch.backends.cudnn.deterministic=True
	torch.use_deterministic_algorithms = True

def load_finished_dataset(file_name):
	tsv_data = pd.read_csv(file_name, sep='\t',header=0)
	finished_list = tsv_data["dataset"].values.tolist()

	return finished_list


# training for a specific model on all the dataset in the given fold.
def evaluate_all(args):

	skip_dataset_list = []
	if args.skip_list != "None":
		skip_dataset_list = load_finished_dataset(args.skip_list)

	cnn_kernel_size = [ int(k) for k in args.cnn_kernel_size.split(',')]
	results = {}

	tfbs = os.listdir(args.data_dir)
	cnn_kernel_size = [ int(k) for k in args.cnn_kernel_size.split(',')]

	print("dataset	AUROC	AUPRC	F1	MCC")

	for i, t_file in enumerate(tfbs):
		task_fold =  args.data_dir + "/" + t_file

		if "wgEncodeAwg" not in t_file: 
			continue
		if t_file in skip_dataset_list:
			continue

		# model train
		model_save_name = train(args.model, task_fold, args.model_dir, args.kmer, cnn_kernel_size, \
			args.embedding, args.embedding_file, \
			args.device, args.batch_size, args.lr, args.epoch, args.dropout, args.num_worker, t_file, False)

		# model evaluation
		
		mcc, auc_score, auprc, f1 = test(args.model, task_fold , model_save_name, args.kmer, cnn_kernel_size, \
			args.embedding, args.embedding_file, args.device, args.dropout, False, args.num_worker)

		results[t_file] = [mcc, auc_score, auprc, f1]

		print("%s\t%f\t%f\t%f\t%f" %(t_file, auc_score, auprc, f1, mcc))

	# saving results to the file

	return results



if __name__ == "__main__":

	set_seed(123)

	parser = argparse.ArgumentParser(description='<TATA Promoter Prediction Task>')

	parser.add_argument('--kmer',default=1,      type=int,  required=True, help="Kmer size for the input.")
	parser.add_argument('--model',     default='deepromoter',   type=str, required=True,  help="DL models [CNN_biLSTM/deepPromoterNet].")
	parser.add_argument('--model_dir', action="store",   type=str, required=True,  help="directory for saving the trained model.")
	parser.add_argument('--device',       default="cpu", type=str, required=False, help='GPU Device(s) used for training')
	
	parser.add_argument('--data_dir', action="store",   type=str, required=True,  help="directory of data.")
	
	parser.add_argument('--lr',     	default=0.001,   type=float, required=False, help='Learning rate')
	parser.add_argument('--epoch',      default=50,       type=int, required=False, help='Training epcohs')
	parser.add_argument('--batch_size' ,default=32,      type=int,  required=False, help="batch_size of the training.")
	parser.add_argument('--dropout'    ,default=0.1,      type=float,  required=False, help="Dropout rate.")

	parser.add_argument('--embedding', default="onehot",  type=str,  required=True, help="The embedding type")
	parser.add_argument('--embedding_file', default="",  type=str,  required=False, help="The embedding file")

	parser.add_argument('--cnn_kernel_size', default="24,17,7", type=str, required=False, help="Kernel size used in CNN.")
	parser.add_argument('--skip_list', default="None", type=str, required=False, help='Skip list')

	parser.add_argument('--num_worker', default=16, type=int, required=False, help='Number of CPU workers for data processing.')


	args = parser.parse_args()

	#file_fold = "/ws1/pretrain/randpt/data/TBFS/motif_occupancy/"

	#print(load_finished_dataset(args.skip_list))

	results = evaluate_all(args)
