import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from torchvision import models
from layers import GraphConvolution
from utils import *


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.classifier = nn.Linear(nfeat, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        text_cls = self.classifier(x[:4576])
        img_cls  = self.classifier(x[4576:])

        return F.log_softmax(x, 0), text_cls, img_cls

#======================================================================================================#
#======================================================================================================#

class Text_Encoder(nn.Module):
	def __init__(self, word_dim=300, hidden_dim=512, text_maxlen=300, cls_cnt=2349):
		super(Text_Encoder, self).__init__()

		self.embedding_dim = hidden_dim

		i2v = np.load('../data/word_emb/gb_i2v.npy')
		self.idx2vec = nn.Embedding(i2v.shape[0]+len(['unk','pad']),
									embedding_dim=word_dim, padding_idx=1)
		self.idx2vec.weight.data[2:].copy_(torch.from_numpy(i2v))
		self.idx2vec.weight.requires_grad = True
		self.unk2vec = nn.Parameter(torch.randn(word_dim))

		self.model = nn.LSTM(word_dim, hidden_dim//2, 2, bidirectional=True, batch_first=True)
		#self.transfer = nn.Conv1d(hidden_dim, hidden_dim, 1)

		self.classifier = nn.Linear(hidden_dim, cls_cnt)

	def word_emb(self, text):
		# input : (batch x length(300))
		text_emb = self.idx2vec(text)
		unks = (text==0).nonzero()
		for (r,c) in unks:
			text_emb[r,c]=self.unk2vec

		return text_emb

	def forward(self, input, input_len):
		# input : (batch x length(300))
		input = self.idx2vec(input)
		# input : (batch x length(300) x dim(300))

		sorted_data, sorted_len, idx_unsort = sort_sequence(input, input_len)
		packed = rnn.pack_padded_sequence(sorted_data, sorted_len, batch_first=True)

		model_out, _ = self.model(packed)

		unpacked, _ = rnn.pad_packed_sequence(model_out, batch_first=True)
		unsorted_data = unsort_sequence(unpacked, idx_unsort, input_len)

		#transfered_data = self.transfer(unsorted_data.unsqueeze(2)).squeeze(2)
		out_cls = self.classifier(unsorted_data)
		return unsorted_data, out_cls

#======================================================================================================#
#======================================================================================================#

class Resnet(nn.Module):
	def __init__(self, embedding_dim, num_typo, img_size):
		super(Resnet, self).__init__()
		self.img_width = img_size[1]
		self.img_height = img_size[0]

		resnet18 =  getattr(models, 'resnet18')(pretrained=False)
		resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
										bias=False)

		self.content_conv = nn.Sequential(
										resnet18.conv1,
										resnet18.bn1,
										resnet18.relu,
										resnet18.maxpool,
										resnet18.layer1,
										resnet18.layer2,
										resnet18.layer3,
										resnet18.layer4,
										)
		# [batch x 512 x 1 x 8]
		self.projector = nn.Conv2d(512, embedding_dim, (1,8))
		self.classifier = nn.Linear(512 * 8, num_typo)

		# Decoder specification
		self.dec_linear = nn.Sequential(
										nn.Linear(512 * 8, self.img_width*self.img_height),
										nn.Sigmoid()
										)

	def forward(self, image):
		res_vec = self.content_conv(image)
		out_vec = self.projector(res_vec).squeeze()
		out_cls = self.classifier(res_vec.view(res_vec.size(0), -1))
		out_img = self.dec_linear(res_vec.view(res_vec.size(0), -1))
		out_img = out_img.view([image.size(0), 1, self.img_width, self.img_height])

		return out_img, out_vec, out_cls
