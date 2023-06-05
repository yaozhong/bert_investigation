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


