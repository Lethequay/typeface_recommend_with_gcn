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
	def __init__(self, config, train_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.test_loader = test_loader

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
		self.image_encoder = Image_Encoder(self.z_dim, self.image_size)
		self.optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, self.text_encoder.parameters())) + \
									list(self.image_encoder.parameters()),
									self.lr, [self.beta1, self.beta2])

		if torch.cuda.is_available():
			self.text_encoder.cuda()
			self.image_encoder.cuda()

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.text_encoder.zero_grad()
		self.image_encoder.zero_grad()

	def train(self):

		iters_per_epoch = len(self.train_loader)
		start_time = time.time()

		for epoch in range(self.num_epochs):
			for i, (typography, pos_image, neg_image, text, text_len) in enumerate(self.train_loader):
				pos_image = pos_image.to(self.device)
				neg_image = neg_image.to(self.device)
				text = text.to(self.device)
				text_len = text_len.to(self.device)

				# (batch x z_dim)
				text_emb  = self.text_encoder(text, text_len)
				# (batch x z_dim), (batch x z_dim)
				pos_style_emb, pos_context_emb = self.image_encoder(pos_image)
				neg_style_emb, neg_context_emb = self.image_encoder(neg_image)

				loss = F.triplet_margin_loss(text_emb, pos_style_emb, neg_style_emb, margin=1)
				# GAN / OCR


				# Backprop + Optimize
				self.reset_grad()
				loss.backward()
				self.optimizer.step()

				# Print the log info
				if (i+1) % self.log_step == 0:
					elapsed = time.time() - start_time
					elapsed = str(datetime.timedelta(seconds=elapsed))

					log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
							elapsed, epoch+1, self.num_epochs, i+1, iters_per_epoch)
					log += ", loss: {:.4f}".format(loss)
					print(log)

			if (epoch+1) % self.val_step == 0:
				self.text_encoder.train(False)
				self.image_encoder.train(False)
				self.text_encoder.eval()
				self.image_encoder.eval()

				val_loss = 0.
				for i, (typography, pos_image,_, text, text_len) in enumerate(self.train_loader):
					image = pos_image.to(self.device)
					text = text.to(self.device)
					text_len = text_len.to(self.device)

					# (batch x z_dim)
					text_emb  = self.text_encoder(text, text_len)
					# (batch x z_dim), (batch x z_dim)
					style_emb, context_emb = self.image_encoder(image)

					val_loss += torch.mean((text_emb - style_emb) ** 2).item()

				print('Valid Loss: ', end='')
				print("{:.4f}".format(val_loss/(i+1)))

				te_path = os.path.join(self.model_path, 'text-encoder-%d.pkl' %(epoch+1))
				ie_path = os.path.join(self.model_path, 'image-encoder-%d.pkl' %(epoch+1))
				torch.save(self.text_encoder.state_dict(), te_path)
				torch.save(self.image_encoder.state_dict(), ie_path)

				self.text_encoder.train(True)
				self.image_encoder.train(True)
