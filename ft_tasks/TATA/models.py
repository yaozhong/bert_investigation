# deep learning models
import torch
import torch.nn as nn
import math


# use the same architure as deepPromoter
## note the difference from the original paper : active function, softSign in the code
## model parameters need to be automatically calucateled according to different input.

class deepPromoterNet(nn.Module):
	def __init__(self, vocab_size, kernel_list= [27, 14, 7], kmer=1, dropout=0.1):
		super(deepPromoterNet, self).__init__()

		self.cnn_diff_kernel = nn.ModuleList()

		for k in kernel_list:
			cnn = nn.Sequential(nn.Conv1d(in_channels=vocab_size, out_channels=vocab_size, kernel_size=k, stride=1, padding="same"),
								 nn.ReLU(True), nn.MaxPool1d(6), nn.Dropout(dropout))
			self.cnn_diff_kernel.append(cnn)

		next_input_size = math.floor((300 - kmer + 1) / 6)
		hidden_size = 32
		self.biLSTM = nn.LSTM(input_size= next_input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
		self.linear = nn.Linear(hidden_size*2, hidden_size)

		self.flatten = nn.Flatten()
		self.fc = nn.Sequential(nn.Linear(vocab_size*3*hidden_size, 128), nn.ReLU(), nn.Dropout(dropout),
			nn.Linear(128, 2), nn.Softmax(dim=1))


	def forward(self, inputs):

		convs = list()
		for cnn in self.cnn_diff_kernel:
			x = cnn(inputs)
			convs.append(x)
		x = torch.cat(convs, 1)

		self.biLSTM.flatten_parameters()

		#print(x.shape)
		x, _ = self.biLSTM(x)
		#print(x.shape)
		x = self.linear(x)
		#print(x.shape)
		x = self.flatten(x)
		y = self.fc(x)

		return y

# using the model from haoyang zeng's work for the tfbs prediction.
class zeng_CNN(nn.Module):
	def __init__(self, feat_size, ksize=[24], kmer=1, dropout=0.1):
		super(zeng_CNN, self).__init__()

		self.cnn_1m = nn.Sequential(nn.Conv1d(in_channels=feat_size, out_channels=128, kernel_size=ksize[0], stride=1, padding="same"),
			nn.ReLU(True),)

		#self.flatten = nn.Flatten()
		self.fc = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(dropout),
								nn.Linear(32, 2), nn.Softmax(dim=1))

	def forward(self, input):

		x = self.cnn_1m(input)
		#x = self.flatten(x)
		#global max pooling
		x, _ = torch.max(x,2)
		y = self.fc(x)

		return y

