# data loading 

import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, Subset
from sklearn.model_selection import train_test_split

from embeddings import *
from functools import partial
import random

random_seed=123

def fix_worker_init_fn(worker_id):
	random.seed(random_seed)

class TBFS_dataset(Dataset):

	def __init__(self, file_name, kmer, filter=None):
		raw_df = pd.read_csv(file_name, header=None, sep=" ", names=('loc', 'seq', 'label'))
		self.number = len(raw_df)

		self.X = raw_df["seq"]
		self.Y = raw_df["label"]
		self.loc = raw_df["loc"]

	def __len__(self):
		return	self.number

	def __getitem__(self, idx):
		return self.X[idx], self.Y[idx], self.loc[idx]

	def get_label_list(self):
		return self.Y


def my_collate_fn(batch, kmer, dic):

	seqs, labels, locs = [],[],[]

	for record in batch:
		if record is not None:
			seq, label, loc = record

			if 'N' in seq:
				# print("@ Remove sample [%s] containing N" %(seq))
				continue

			seqs.append(seqs2kmerVec(seq, kmer, dic))
			labels.append(label)
			locs.append(loc)

	seqs = np.array(seqs)
	return seqs, labels, locs


