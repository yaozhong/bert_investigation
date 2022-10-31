# deep learning models
import torch
import torch.nn as nn
import math

# embedding non-linear transform
class CNN_biLSTM_et(nn.Module):
	def __init__(self, feat_size, ksize=[7,7,7], kmer=1, dropout=0.1):
		super(CNN_biLSTM_et, self).__init__()

		self.embed_fc = nn.Sequential(nn.Linear(feat_size, 100), nn.ReLU())

		self.cnn_3m = nn.Sequential(nn.Conv1d(in_channels=100, out_channels=64, kernel_size=ksize[0], stride=1, padding="same"),
			nn.ReLU(True), nn.MaxPool1d(2),
			nn.Conv1d(in_channels=64, out_channels=128, kernel_size=ksize[1], stride=1, padding="same"),
			nn.ReLU(True), nn.MaxPool1d(2),
			nn.Conv1d(in_channels=128, out_channels=256, kernel_size=ksize[2], stride=1, padding="same"),
			nn.ReLU(True), nn.MaxPool1d(2))

		self.BiLSTM = nn.LSTM(input_size=37, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)

		self.linear = nn.Linear(32*2, 32)
		self.flatten = nn.Flatten()
		
		self.fc = nn.Sequential(nn.Linear(8192, 32), nn.ReLU(), nn.Dropout(dropout),
								nn.Linear(32, 2), nn.Softmax(dim=1))

	def forward(self, input):
		x = self.embed_fc(input.transpose(2,1))
		x = self.cnn_3m(x.transpose(2,1))

		self.BiLSTM.flatten_parameters()		
		out, _ = self.BiLSTM(x)

		out = self.linear(out)
		out = self.flatten(out)
		
		y = self.fc(out)

		return y


## fixed embedding, currently tested
class CNN_biLSTM(nn.Module):
	def __init__(self, feat_size, ksize=[7,7,7], kmer=1, dropout=0.1):
		super(CNN_biLSTM, self).__init__()

		self.cnn_3m = nn.Sequential(nn.Conv1d(in_channels=feat_size, out_channels=64, kernel_size=ksize[0], stride=1, padding="same"),
			nn.ReLU(True), nn.MaxPool1d(2),
			nn.Conv1d(in_channels=64, out_channels=128, kernel_size=ksize[1], stride=1, padding="same"),
			nn.ReLU(True), nn.MaxPool1d(2),
			nn.Conv1d(in_channels=128, out_channels=256, kernel_size=ksize[2], stride=1, padding="same"),
			nn.ReLU(True), nn.MaxPool1d(2))

		self.BiLSTM = nn.LSTM(input_size=37, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)

		self.linear = nn.Linear(32*2, 32)
		self.flatten = nn.Flatten()
		
		self.fc = nn.Sequential(nn.Linear(8192, 32), nn.ReLU(), nn.Dropout(dropout),
								nn.Linear(32, 2), nn.Softmax(dim=1))

	def forward(self, input):
		x = self.cnn_3m(input)

		self.BiLSTM.flatten_parameters()		
		out, _ = self.BiLSTM(x)

		out = self.linear(out)
		out = self.flatten(out)
		
		y = self.fc(out)

		return y



class CNN_biLSTM_ht(nn.Module):
	def __init__(self, feat_size, ksize=[7,7,7], kmer=1, dropout=0.1, h_fc=32):
		super(CNN_biLSTM_ht, self).__init__()

		self.cnn_3m = nn.Sequential(nn.Conv1d(in_channels=feat_size, out_channels=64, kernel_size=ksize[0], stride=1, padding="same"),
			nn.ReLU(True), nn.MaxPool1d(2),
			nn.Conv1d(in_channels=64, out_channels=128, kernel_size=ksize[1], stride=1, padding="same"),
			nn.ReLU(True), nn.MaxPool1d(2),
			nn.Conv1d(in_channels=128, out_channels=256, kernel_size=ksize[2], stride=1, padding="same"),
			nn.ReLU(True), nn.MaxPool1d(2))

		self.BiLSTM = nn.LSTM(input_size=37, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)

		self.linear = nn.Linear(32*2, 32)
		self.flatten = nn.Flatten()
		
		self.fc = nn.Sequential(nn.Linear(8192, h_fc), nn.ReLU(), nn.Dropout(dropout),
								nn.Linear(h_fc, 2), nn.Softmax(dim=1))

	def forward(self, input):
		x = self.cnn_3m(input)

		self.BiLSTM.flatten_parameters()		
		out, _ = self.BiLSTM(x)

		out = self.linear(out)
		out = self.flatten(out)
		
		y = self.fc(out)

		return y


class CNN_biLSTM_test(nn.Module):
	def __init__(self, feat_size, ksize=[11,17,17], kmer=1, dropout=0.1):
		super(CNN_biLSTM_test, self).__init__()

		self.cnn_3m = nn.Sequential(nn.Conv1d(in_channels=feat_size, out_channels=16, kernel_size=ksize[0], stride=1, padding="valid"),
			nn.ReLU(True), nn.MaxPool1d(2),
			nn.Conv1d(in_channels=16, out_channels=32, kernel_size=ksize[1], stride=1, padding="valid"),
			nn.ReLU(True), nn.MaxPool1d(2),
			nn.Conv1d(in_channels=32, out_channels=64, kernel_size=ksize[2], stride=1, padding="valid"),
			nn.ReLU(True), nn.MaxPool1d(2))

		self.BiLSTM = nn.LSTM(input_size=34, hidden_size=32, num_layers=1, batch_first=True, bidirectional=False)

		self.linear = nn.Linear(32, 32)
		self.flatten = nn.Flatten()
		
		self.fc = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(dropout),
								nn.Linear(128, 2), nn.Softmax(dim=1))

	def forward(self, input):
		x = self.cnn_3m(input)

		self.BiLSTM.flatten_parameters()		
		out, _ = self.BiLSTM(x)

		out = self.linear(out)
		out = self.flatten(out)
		
		y = self.fc(out)

		return y

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


# 3 layers
class CNN_test(nn.Module):
	def __init__(self, feat_size, ksize=[5, 5, 5], kmer=1, dropout=0.1):
		super(CNN_test, self).__init__()

		self.cnn_3m = nn.Sequential(nn.Conv1d(in_channels=feat_size, out_channels=16, kernel_size=ksize[0], stride=1, padding="valid"),
			nn.ReLU(True), nn.MaxPool1d(2),
			nn.Conv1d(in_channels=16, out_channels=32, kernel_size=ksize[1], stride=1, padding="valid"),
			nn.ReLU(True), nn.MaxPool1d(2),
			nn.Conv1d(in_channels=32, out_channels=64, kernel_size=ksize[2], stride=1, padding="valid"),
			nn.ReLU(True), nn.MaxPool1d(2))

		self.flatten = nn.Flatten()
		self.fc = nn.Sequential(nn.Linear(2112, 32), nn.ReLU(), nn.Dropout(dropout),
								nn.Linear(32, 2), nn.Softmax(dim=1))

	def forward(self, input):

		x = self.cnn_3m(input)
		x = self.flatten(x)
		#global max pooling
		#x, _ = torch.max(x,2)
		y = self.fc(x)

		return y


#########################################
##------------- testing ---------------##
#########################################

class deepPromoterNet_test(nn.Module):
	def __init__(self, vocab_size, kernel_list= [27, 14, 7], kmer=1, dropout=0.1):
		super(deepPromoterNet_test, self).__init__()

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
		input_size = vocab_size * 3*hidden_size #vocab_size*3*hidden_size
		self.fc = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(dropout),
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
