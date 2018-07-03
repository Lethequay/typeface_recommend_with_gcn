from collections import namedtuple
import numpy as np
import functools
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import torchvision
from torchvision import models
from utils import *


class Text_Encoder(nn.Module):
	def __init__(self, embedding_size=300, text_maxlen=300):
		super(Text_Encoder, self).__init__()

		self.embedding_size = embedding_size

		ft_i2v = np.load('./data/word_emb/ft_i2v.npy')
		self.idx2vec = nn.Embedding(ft_i2v.shape[0]+len(['unk','pad']),
									embedding_dim=embedding_size, padding_idx=1)
		self.idx2vec.weight.data[2:].copy_(torch.from_numpy(ft_i2v))

		self.model = nn.LSTM(embedding_size, embedding_size, 2, bidirectional=True)

	def forward(self, input, input_len):
		# batch x channels(300) x length(300)
		input = self.idx2vec(input)

		sorted_data, sorted_len, idx_unsort = sort_sequence(input, input_len)
		packed = rnn.pack_padded_sequence(sorted_data, sorted_len, batch_first=True)

		model_out, _ = self.model(packed)

		unpacked, _ = rnn.pad_packed_sequence(model_out, batch_first=True)
		return unsort_sequence(unpacked, idx_unsort)[:, -1, :]

class Image_Encoder(nn.Module):
	def __init__(self, image_size):
		super(Image_Encoder, self).__init__()

		self.image_size = image_size
		self.kernel_sizes = range(5,15)
		self.convs = nn.ModuleList([nn.Sequential(
					 nn.Conv2d(2, 10, kernel_size=i),
					 nn.ReLU(inplace=True))
					 for i in self.kernel_sizes])
		self.avgpools = nn.ModuleList([
						nn.AvgPool2d((self.image_size[1]-i+1, self.image_size[0]-i+1))
						for i in self.kernel_sizes])
		self.maxpools = nn.ModuleList([
						nn.MaxPool2d((self.image_size[1]-i+1, self.image_size[0]-i+1))
						for i in self.kernel_sizes])


	def forward(self, input, typography):
		# input : (440 x 231)
		for i, k in enumerate(self.kernel_sizes):
			conv_out = self.convs[i](input)
			# mean/max avg
			maxpool_out = self.maxpools[i](conv_out)
			avgpool_out = self.avgpools[i](conv_out)
			print(maxpool_out.size())
			print(avgpool_out.size())
			break
