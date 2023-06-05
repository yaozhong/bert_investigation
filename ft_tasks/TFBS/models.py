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

