# TATA data generation

from pyfaidx import Fasta
import random
import numpy as np
from itertools import product

def negative_gen_as_deepromoter(seq, subunit_len=15, shuffle_number=12):
	num_seg = int(len(seq)/subunit_len)
	shuffle_segs = random.sample(range(num_seg), shuffle_number)

	neg_seq = ""
	for i in range(num_seg):
		if i in shuffle_segs:
			sub_seq = "".join(np.random.choice(['A','T','G','C'], subunit_len))	
		else:
			sub_seq = seq[i*subunit_len:(i+1)*subunit_len]

		neg_seq = neg_seq + sub_seq

	return neg_seq

""" 
may contains non-TATA sequences 
using 
"""
def search_tata_in_genome(ref_genome, promoter_df):
    """
    ref to : https://github.com/joanaapa/Distillation-DNABERT-Promoter/blob/main/data/promoters/get_promoter.py
    """
    promoter_df = promoter_df[["Chr", "TSS"]]
    coord = []
    # For each chromosome, take sequences in-between promoters and find TATA sequences
    for i in range(1, 23):
        chrx = ref_genome["chr" + str(i)][:].seq
        tss_chr = [0] + promoter_df[promoter_df["Chr"] == "chr" + str(i)]["TSS"].values.tolist()
        finds = []
        for j in range(len(tss_chr) - 1):
            seq = chrx[tss_chr[j] + 50 : tss_chr[j + 1] - 250]
            if len(seq) > 500:
                new_finds = [m.start(0) + tss_chr[j] + 50 for m in re.finditer("tata", seq, re.I)]
                finds = finds + new_finds
        coord += [list(x) for x in zip(["chr" + str(i)] * len(finds), finds)]
    return coord


def extract_seq(ref_genome, chrx, start, end):
    return ref_genome[chrx][start:end].seq.upper()


def 



def negative_gen4TATA(seq):
	neg_seq = ""
	count = 0
	if "TATA" not in seq:
		count = 1
		
	return count


def generate_dataset_from_fa(fa_file, dev_test_per=0.1, tag="noTATA"):

	p_set, n_set = [], []

	regions = Fasta(fa_file)
	count = 0
	print(len(regions.keys()))
	for r in regions.keys():
		seq = str(regions[r])
		p_set.append(seq)
		if tag != "TATA":
			neg_seq = negative_gen_as_deepromoter(seq, 15, 12)
			n_set.append(neg_seq)

	# generate negative samples independently
	if tag == "TATA":
		print("Please generate negative TATA samples independently")

	#random shuffle data and generate the data.
	random.shuffle(p_set)
	random.shuffle(n_set)

	train_split_idx = int(len(p_set) * (1-dev_test_per*2))
    dev_idx = train_split_idx + int(len(p_set)*dev_test_per)

    # write files 





if __name__ == "__main__":

	seed = 123

	random.seed(seed)
	np.random.seed(seed)

	fa_data_path = "/ws1/pretrain/randpt/data/TATA/EPD_raw/"
	fa_file = fa_data_path + "hg38_human_TATA_3065.fa"

	generate_dataset_from_fa(fa_file, "TATA")
