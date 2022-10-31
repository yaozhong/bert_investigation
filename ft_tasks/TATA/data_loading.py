# data loading 

import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from embeddings import *
from functools import partial
import random

random_seed=123

def fix_worker_init_fn(worker_id):
	random.seed(random_seed)

class TATA_dataset(Dataset):

	def __init__(self, file_name, kmer, filter=None):
		raw_df = pd.read_csv(file_name, header=0, sep="\t")
		self.number = len(raw_df)

		self.X = raw_df["sequence"]
		self.Y = raw_df["label"]
		self.category = raw_df["category"]

	def __len__(self):
		return	self.number

	def __getitem__(self, idx):
		return self.X[idx], self.Y[idx], self.category[idx]


def my_collate_fn(batch, kmer, dic):

	seqs, labels, categories = [],[],[]

	for record in batch:
		if record is not None:
			seq, label, category = record

			if 'N' in seq:
				# print("@ Remove sample [%s] containing N" %(seq))
				continue

			seqs.append(seqs2kmerVec(seq, kmer, dic))
			labels.append(label)
			categories.append(category)

	seqs = np.array(seqs)
	return seqs, labels, categories



if __name__ == "__main__":

	batch_size = 128

	dev_file_name = "/ws1/pretrain/randpt/data/TATA/TATA_human_0/dev.tsv"
	train_file_name = "/ws1/pretrain/randpt/data/TATA/TATA_human_0/train.tsv"

	kmer = 2
	dic = get_1hot_embedding_dic(kmer)

	dev_dataset = TATA_dataset(dev_file_name, kmer, dic)
	train_dataset = TATA_dataset(train_file_name, kmer, dic)

	# if do CV, concate those two 
	# whole_train = ConcatDataset([train_dataset, dev_dataset])

	train_ds_generator = DataLoader(train_dataset, batch_size, collate_fn=partial(my_collate_fn,kmer=kmer, dic=dic), num_workers=72) 
	dev_ds_generator = DataLoader(dev_dataset, batch_size, collate_fn=partial(my_collate_fn,kmer=kmer, dic=dic), num_workers=72)


	for x, y, category in train_ds_generator:
		print(y, category)
		print(x.shape)

## local file testing
def test():
	# file
	file_name = "/Users/yaozhong/Desktop/randpt/data/TATA/TATA_human_0/dev.tsv"
	raw_df = pd.read_csv(file_name, header=0, sep="\t")

	kmer = 4
	dic = get_1hot_embedding_dic(kmer)

	x = seqs2kmerVec(raw_df["sequence"], kmer, dic)
	print(len(x))
	print(x[0].shape)

	# prepare embeding mapping dicionary according to the given inputs

	# dataLoader generating the datasets