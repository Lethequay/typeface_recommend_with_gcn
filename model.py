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
	def __init__(self, word_dim=300, hidden_dim=512, text_maxlen=300):
		super(Text_Encoder, self).__init__()

		self.embedding_dim = hidden_dim

		ft_i2v = np.load('./data/word_emb/ft_i2v.npy')
		self.idx2vec = nn.Embedding(ft_i2v.shape[0]+len(['unk','pad']),
									embedding_dim=word_dim, padding_idx=1)
		self.idx2vec.weight.data[2:].copy_(torch.from_numpy(ft_i2v))
		self.idx2vec.weight.requires_grad = True
		self.unk2vec = nn.Parameter(torch.randn(word_dim))

		self.model = nn.LSTM(word_dim, hidden_dim, 2, bidirectional=False)

	def word_emb(self, text):
		# input : (batch x length(300))
		text_emb = self.idx2vec(text)
		unks = (text==0).nonzero()
		for (r,c) in unks:
			text_emb[r,c]=self.unk2vec

		# text_emb : (batch x channels(300) x length(300))
		return text_emb

	def forward(self, input, input_len):
		# input : (batch x length(300))
		input = self.word_emb(input)

		sorted_data, sorted_len, idx_unsort = sort_sequence(input, input_len)
		packed = rnn.pack_padded_sequence(sorted_data, sorted_len, batch_first=True)

		model_out, _ = self.model(packed)

		unpacked, _ = rnn.pad_packed_sequence(model_out, batch_first=True)
		unsorted_data = unsort_sequence(unpacked, idx_unsort)

		return unsorted_data[:,-1,:]

#======================================================================================================#
#======================================================================================================#

class Image_Encoder(nn.Module):
	def __init__(self, embedding_dim, image_size, num_typo):
		super(Image_Encoder, self).__init__()

		self.embedding_dim = embedding_dim
		self.image_size = image_size
		self.num_typo = num_typo
		self.kernel_sizes = range(5,20)
		self.style_convs = nn.ModuleList([nn.Sequential(
						   nn.Conv2d(1, 10, kernel_size=i),
						   nn.ReLU(inplace=True))
						   for i in self.kernel_sizes])
		self.avgpools = nn.ModuleList([
						nn.AvgPool2d((self.image_size[1]-i+1, self.image_size[0]-i+1))
						for i in self.kernel_sizes])
		self.maxpools = nn.ModuleList([
						nn.MaxPool2d((self.image_size[1]-i+1, self.image_size[0]-i+1))
						for i in self.kernel_sizes])

		self.classifier = nn.Linear(self.embedding_dim, self.num_typo)

		self.resnet18 =  getattr(models, 'resnet18')(pretrained=True)
		self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
										bias=False)

		self.context_conv = nn.Sequential(
										self.resnet18.conv1,
										self.resnet18.bn1,
										self.resnet18.relu,
										self.resnet18.maxpool,
										self.resnet18.layer1,
										self.resnet18.layer2
										)
		self.context_rnn = nn.GRU(128*4, self.embedding_dim//2, num_layers=2,
								  batch_first=True, bidirectional=True)

	def forward(self, input):
		# input : (batch x channel(1) x h(196) x w(1104))
		#====================STYLE======================#
		out_list = []
		for i, k in enumerate(self.kernel_sizes):
			conv_out = self.style_convs[i](input)
			# mean/max avg
			maxpool_out = self.maxpools[i](conv_out)
			avgpool_out = self.avgpools[i](conv_out)
			out_list.append(maxpool_out.squeeze())
			out_list.append(avgpool_out.squeeze())
		# style_out : (batch x 300)
		style_out = torch.cat(out_list, 1)
		cls_out = self.classifier(style_out)

		#===================CONTEXT=====================#
		# (batch x 128 x 4 x 32)
		conv_out = self.context_conv(input)
		# (batch x 32 x (128*4))
		seq_out = features_to_sequence(conv_out)
		# (batch, 32, 300), (batch, 16, 150)
		context_out,_ = self.context_rnn(seq_out)

		return style_out, cls_out, context_out[:,-1,:]
