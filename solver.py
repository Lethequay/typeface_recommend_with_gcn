import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from model import *

class Solver(object):
	def __init__(self, config, data_loader):

		# Data loader
		self.data_loader = data_loader

		# Models
		self.text_encoder = None
		self.image_encoder = None

		# Models hyper-parameters
		self.image_size = config.image_size
		self.z_dim = config.z_dim

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2
		self.text_maxlen = config.text_maxlen

		# Training settings
		self.sample_epochs = config.sample_epochs
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.sample_path = config.sample_path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.build_model()

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def build_model(self):
		self.text_encoder = Text_Encoder(self.z_dim, self.text_maxlen)
		self.image_encoder = Image_Encoder(self.image_size)
		self.optimizer = optim.Adam(self.text_encoder.parameters(),
									self.lr, [self.beta1, self.beta2])

		if torch.cuda.is_available():
			self.text_encoder.cuda()
			self.image_encoder.cuda()

	def train(self):

		for i, (typography, image, text, text_len) in enumerate(self.data_loader):
			image = image.to(self.device)
			text = text.to(self.device)
			text_len = text_len.to(self.device)

			text_emb  = self.text_encoder(text, text_len)
			image_emb = self.image_encoder(image, typography)
			break
