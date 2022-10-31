import argparse, random
import string
from itertools import islice

def get_random_string(length):
	letters = ["A", "T", "G", "C"]
	return (''.join(random.choice(letters) for _ in range(length)))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='<plot kmer word embedding>')
	parser.add_argument('-n', default=1000000, type=int, required=True, help="random seq len")
	parser.add_argument('-seed', default=123, type=int, help="random seed for the generation")
	args = parser.parse_args()
	random.seed(args.seed)
	print(get_random_string(args.n))

