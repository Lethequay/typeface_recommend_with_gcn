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
		self.word_dim = config.word_dim
		self.z_dim = config.z_dim
		self.num_typo = config.num_typo

		# Hyper-parameters
		self.lambda_cls = config.lambda_cls
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2
		self.text_maxlen = config.text_maxlen

		# Training settings
		self.start_epochs = config.start_epochs
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
		self.text_encoder = Text_Encoder(self.word_dim, self.z_dim, self.text_maxlen)
		self.image_encoder = Image_Encoder(self.z_dim, self.image_size, self.num_typo)
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

	def restore_model(self, resume_iters):
		"""Restore the trained generator and discriminator."""
		te_path = os.path.join(self.model_path, 'text-encoder-%d.pkl' %(resume_iters))
		ie_path = os.path.join(self.model_path, 'image-encoder-%d.pkl' %(resume_iters))
		if os.path.isfile(te_path) and os.path.isfile(ie_path):
			self.text_encoder.load_state_dict(torch.load(te_path, map_location=lambda storage, loc: storage))
			self.image_encoder.load_state_dict(torch.load(ie_path, map_location=lambda storage, loc: storage))
			print('te/se is Successfully Loaded from %d epoch'%resume_iters)
			return resume_iters
		else:
			return 0


	def train(self):

		start_iter = self.restore_model(self.start_epochs)
		iters_per_epoch = len(self.train_loader)
		start_time = time.time()

		for epoch in range(start_iter, self.num_epochs):
			acc = 0.
			for i, (typography, pos_image, neg_image, text, text_len) in enumerate(self.train_loader):
				pos_image = pos_image.to(self.device)
				neg_image = neg_image.to(self.device)
				text = text.to(self.device)
				text_len = text_len.to(self.device)

				# (batch x z_dim)
				text_emb  = self.text_encoder(text, text_len)
				# (batch x z_dim), (batch x z_dim)
				pos_style_emb, cls_out, pos_content_emb = self.image_encoder(pos_image)
				neg_style_emb, _,       neg_content_emb = self.image_encoder(neg_image)

				_, pred = torch.sort(cls_out, 1, descending=True)
				acc  += precision_at_k(pred.data.cpu().numpy(), typography)
				loss_triplet = F.triplet_margin_loss(text_emb, pos_style_emb, neg_style_emb, margin=1)
				loss_cls = F.cross_entropy(cls_out, typography.to(self.device))
				# GAN / OCR


				# Compute gradient penalty
				loss = loss_triplet + self.lambda_cls * loss_cls

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
					log += ", loss: {:.4f}, precision@5: {:.4f}".format(loss, acc/(i+1))
					print(log)


			#==================================== Validation ====================================#
			if (epoch+1) % self.val_step == 0:
				self.text_encoder.train(False)
				self.image_encoder.train(False)
				self.text_encoder.eval()
				self.image_encoder.eval()

				val_loss = 0.
				val_acc  = 0.
				for i, (typography, pos_image,_, text, text_len) in enumerate(self.test_loader):
					image = pos_image.to(self.device)
					text = text.to(self.device)
					text_len = text_len.to(self.device)

					# (batch x z_dim)
					text_emb  = self.text_encoder(text, text_len)
					# (batch x z_dim), (batch x z_dim)
					style_emb, cls_out, content_emb = self.image_encoder(image)

					_, pred = torch.sort(cls_out, 1, descending=True)
					val_acc  += precision_at_k(pred.data.cpu().numpy(), typography)
					val_loss += torch.mean((text_emb - style_emb) ** 2).item()

				print('Valid Loss: {:.4f}, Valid Precision@5: {:.4f}'\
					  .format(val_loss/(i+1), val_acc/(i+1)))

				te_path = os.path.join(self.model_path, 'text-encoder-%d.pkl' %(epoch+1))
				ie_path = os.path.join(self.model_path, 'image-encoder-%d.pkl' %(epoch+1))
				torch.save(self.text_encoder.state_dict(), te_path)
				torch.save(self.image_encoder.state_dict(), ie_path)

				self.text_encoder.train(True)
				self.image_encoder.train(True)

	def sample(self):

		self.restore_model(self.sample_epochs)
		id2word = np.load("./data/word_emb/ft_i2w.npy")
		word2id = np.load("./data/word_emb/ft_w2i.npy").item()
		typo_list = np.load('./data/typo_list.npy')

		cl_typo = None
		cl_dist = 9999
		from_word = 'digital'
		from_word = Variable(torch.from_numpy(np.asarray(word2id[from_word]))).cuda()
		with torch.no_grad():
			from_word_vec = self.text_encoder.idx2vec(from_word)
			for i, (typography, pos_image,_, text,_) in enumerate(self.test_loader):
				pos_image = pos_image.to(self.device)
				pos_style_emb, _, pos_content_emb = self.image_encoder(pos_image)

				# The Closest Words from a Typography
				# (batch x 999994)
				mm = torch.mm(pos_style_emb, self.text_encoder.idx2vec.weight[2:].t())

				batch_size = 10
				iters = mm.size(0)//batch_size + 1
				for j in range(iters):
					_, rank = torch.sort(mm[j*batch_size:(j+1)*batch_size], descending=True)
					rank = rank.data.cpu().numpy()
					for k in range(batch_size):
						print(typo_list[typography[i*batch_size+k].numpy()],
							  [id2word[idx] for idx in rank[k,:10]])
					break
				break
				'''

				# The Closest Typographies from a Word
				# (1 x batch_size)
				mm = torch.abs(torch.mm(from_word_vec.unsqueeze(0), pos_style_emb.t()))
				min_value, min_idx = torch.min(mm, 1)
				if cl_dist > min_value[0]:
					cl_dist = min_value[0]
					cl_typo = typo_list[typography[min_idx[0]].numpy()]
					print(cl_dist.cpu().numpy(), cl_typo)
				'''

			print("The Closest Typo is", cl_typo)
