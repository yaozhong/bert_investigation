# deep learning models
import torch
import torch.nn as nn
import math

## fixed embedding
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


## DNAQ MODEL
"""
class DanQ(nn.Module):
    def __init__(self, feat_size, ksize=[24], kmer=1, dropout=0.1):
        super(DanQ, self).__init__()

        self.cnn_1m = nn.Sequential(nn.Conv1d(in_channels=feat_size, out_channels=320, kernel_size=ksize[0], stride=1, padding="same"),
			nn.ReLU(True),nn.MaxPool1d(kernel_size=13, stride=13), nn.Dropout(dropout))

        self.BiLSTM = nn.LSTM(input_size=320, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(nn.Linear(8192, 919), nn.ReLU(), nn.Dropout(dropout),
								nn.Linear(32, 2), nn.Softmax(dim=1))
      
        self.Linear1 = nn.Linear(75*640, 925)
        self.Linear2 = nn.Linear(925, 919)

    def forward(self, input):
"""        

## currently tunning
class CNN_biLSTM_tune(nn.Module):
	def __init__(self, feat_size, ksize=[24], kmer=1, dropout=0.1):
		super(CNN_biLSTM_tune, self).__init__()

		self.cnn_3m = nn.Sequential(nn.Conv1d(in_channels=feat_size, out_channels=128, kernel_size=ksize[0], stride=1, padding="same"),
			nn.ReLU(True), nn.MaxPool1d(2),)

		self.BiLSTM = nn.LSTM(input_size=48, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)

		self.linear = nn.Linear(32*2, 32)
		self.flatten = nn.Flatten()
		
		self.fc = nn.Sequential(nn.Linear(4096, 32), nn.ReLU(), nn.Dropout(dropout),
								nn.Linear(32, 2), nn.Softmax(dim=1))

	def forward(self, input):
		x = self.cnn_3m(input)

		self.BiLSTM.flatten_parameters()		
		out, _ = self.BiLSTM(x)

		out = self.linear(out)
		out = self.flatten(out)
		
		y = self.fc(out)

		return y



class CNN_biLSTM(nn.Module):
	def __init__(self, feat_size, ksize=[7,7,7], kmer=1, dropout=0.1):
		super(CNN_biLSTM, self).__init__()

		self.cnn_3m = nn.Sequential(nn.Conv1d(in_channels=feat_size, out_channels=64, kernel_size=ksize[0], stride=1, padding="same"),
			nn.ReLU(True), nn.MaxPool1d(2),
			nn.Conv1d(in_channels=64, out_channels=128, kernel_size=ksize[1], stride=1, padding="same"),
			nn.ReLU(True), nn.MaxPool1d(2),
			nn.Conv1d(in_channels=128, out_channels=256, kernel_size=ksize[2], stride=1, padding="same"),
			nn.ReLU(True), nn.MaxPool1d(2))

		self.BiLSTM = nn.LSTM(input_size=12, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)

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

		next_input_size = math.floor((100 - kmer + 1) / 6)
		self.biLSTM = nn.LSTM(input_size= next_input_size, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
		self.linear = nn.Linear(32*2, 32)

		self.flatten = nn.Flatten()
		self.fc = nn.Sequential(nn.Linear(vocab_size*3*32, 128), nn.ReLU(), nn.Dropout(dropout),
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


### models used for hyper-parameters tunning
class CNN_biLSTM_ht(nn.Module):
	def __init__(self, feat_size, ksize=[7,7,7], kmer=1, dropout=0.1, h_fc=32):
		super(CNN_biLSTM_ht, self).__init__()

		self.cnn_3m = nn.Sequential(nn.Conv1d(in_channels=feat_size, out_channels=64, kernel_size=ksize[0], stride=1, padding="same"),
			nn.ReLU(True), nn.MaxPool1d(2),
			nn.Conv1d(in_channels=64, out_channels=128, kernel_size=ksize[1], stride=1, padding="same"),
			nn.ReLU(True), nn.MaxPool1d(2),
			nn.Conv1d(in_channels=128, out_channels=256, kernel_size=ksize[2], stride=1, padding="same"),
			nn.ReLU(True), nn.MaxPool1d(2))

		self.BiLSTM = nn.LSTM(input_size=12, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)

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
