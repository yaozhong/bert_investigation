# embeddings

import numpy as np
from itertools import product


def load_vocab(file):

	vocab_file = open(file, 'r')
	Lines = vocab_file.readlines()

	vocab_list = []
	for kmer in Lines:
		vocab_list.append(kmer.strip() )

	vocab_file.close()
	return vocab_list

# gpn embedding
#def get_gpn_embedding_dic(kmer):

	


# embedding dictionary 
## loading w2v
def get_dna2vec_embedding_dic(filepath, kmer):

	from dna2vec.multi_k_model import MultiKModel

	kmer_list = ["".join(x) for x in list(product('ATGC', repeat=kmer))]
	mk_model = MultiKModel(filepath)
	DNA2Vec = {}

	for kmer in kmer_list:
		vec = mk_model.vector(kmer)
		DNA2Vec[kmer] = vec
	
	return DNA2Vec

## loading 1-hot embedding
def get_1hot_embedding_dic(kmer):

	kmer_list = ["".join(x) for x in list(product('ATGC', repeat=kmer))]
	DNA2Vec = {}

	for i,kmer in enumerate(kmer_list):
		#vec = mk_model.vector(kmer)
		vec = np.zeros(len(kmer_list))
		vec[i] = 1
		DNA2Vec[kmer] = (vec)
	
	return DNA2Vec


## dnaBERT
def get_bert_embedding(model_file):

	from transformers import BertModel

	model = BertModel.from_pretrained(model_file, output_hidden_states = True).to("cpu")
	embeddings = model.embeddings.word_embeddings.weight
	embeddings = embeddings.detach().numpy()

	
	position_embeddings = model.embeddings.position_embeddings.weight
	position_embeddings = position_embeddings.detach().numpy()

	vocab_file = model_file + "/vocab.txt"
	vocab = load_vocab(vocab_file)

	DNA2Vec = {}
	for i, v in enumerate(vocab):
		DNA2Vec[v] = embeddings[i]

	return DNA2Vec, position_embeddings


################################################

def seqs2kmer(seqs, kmer):
	seq_kmers = []
	for s in seqs:
		seq_kmers.append([s[i:i+kmer] for i in range(0, len(s)-kmer+1)])
	return seq_kmers


def seqs2kmerVec(seq, kmer, dic):
	seqs_embed = []

	#print([s[i:i+kmer] for i in range(0, len(s)-kmer+1)])
	embed = [dic[seq[i:i+kmer]] for i in range(0, len(seq)-kmer+1)]
	seqs_embed.append(embed)

	return np.vstack(seqs_embed)

