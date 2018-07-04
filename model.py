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
		self.kernel_sizes = range(5,20)
		self.style_convs = nn.ModuleList([nn.Sequential(
						   nn.Conv2d(2, 10, kernel_size=i),
						   nn.ReLU(inplace=True))
						   for i in self.kernel_sizes])

		self.resnet18 =  getattr(models, 'resnet18')(pretrained=True)
		self.resnet18.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
		                                bias=False)
		self.context_conv = nn.Sequential(
							            self.resnet18.conv1,
							            self.resnet18.bn1,
							            self.resnet18.relu,
							            self.resnet18.maxpool,
							            self.resnet18.layer1,
							            self.resnet18.layer2,
							            self.resnet18.layer3,
							            self.resnet18.layer4
								        )
		#context_rnn = nn.GRU()
		self.avgpools = nn.ModuleList([
						nn.AvgPool2d((self.image_size[1]-i+1, self.image_size[0]-i+1))
						for i in self.kernel_sizes])
		self.maxpools = nn.ModuleList([
						nn.MaxPool2d((self.image_size[1]-i+1, self.image_size[0]-i+1))
						for i in self.kernel_sizes])


	def forward(self, input, typography):

		conv_out = self.context_conv(input)
		print(conv_out.size())
		seq_out = features_to_sequence(conv_out)
		print(seq_out.size())


		# input : (440 x 231)
		out_list = []
		for i, k in enumerate(self.kernel_sizes):
			conv_out = self.convs[i](input)
			# mean/max avg
			maxpool_out = self.maxpools[i](conv_out)
			avgpool_out = self.avgpools[i](conv_out)
			out_list.append(maxpool_out.squeeze())
			out_list.append(avgpool_out.squeeze())
		# style_out : (batch x 300)
		style_out = torch.cat(out_list, 1)


		return style_out
