#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image, ImageOps, ImageFont, ImageDraw

class ImageFolder(data.Dataset):
	"""Load Variaty Chinese Fonts for Iterator. """
	def __init__(self, mode, image_path, image_size, transform=None):
		"""Initializes image paths and preprocessing module."""
		self.mode = mode

		self.idx2text = np.load('../data/idx2text.npy')
		self.idx2typos = np.load('../data/idx2typos.npy')
		typo_list = np.load('../data/typo_list.npy')
		self.typo_cnt = len(typo_list)

		self.image_size = image_size
		self.image_path = image_path
		self.image_paths = list(map(lambda x: os.path.join(image_path, x), os.listdir(image_path)))

		self.transform = transform
		self.train_dataset = []
		self.test_dataset = []
		self.preprocess()

		if mode == 'train':
			self.data_size = len(self.train_dataset)
		else:
			self.data_size = len(self.test_dataset)
		print("typo count :", self.typo_cnt)
		print("text count :", len(self.image_paths))
		print("data count :", self.data_size)

	def preprocess(self):
		train_thr = int(len(self.idx2text) * 0.8)
		for i, data in enumerate(self.idx2text):
			if (i+1) < train_thr:
				self.train_dataset.append(data)
			else:
				self.test_dataset.append(data)

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		data = self.train_dataset if self.mode == 'train' else self.test_dataset
		typos= self.idx2typos[index]
		text = self.idx2text[index]
		length = int(text[-1])
		text = text[:-1]
		text = torch.from_numpy(np.asarray(text))

		image_set = []
		for typography in typos:
			pos_image = Image.open(self.image_path+str(typography)+'.png').convert("L")
			pos_image = pos_image.resize(self.image_size, Image.ANTIALIAS)
			if self.transform is not None:
				pos_image = self.transform(pos_image)
			print(pos_image)
			image_set.append(pos_image)
		text_typo_label = torch.from_numpy(np.asarray(
						  [1 if i in typos else 0 for i in range(self.typo_cnt)]))

		return typography, image_set, text, length, text_typo_label

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.idx2text)


def get_loader(mode, image_path, image_size, batch_size, num_workers=2):
	"""Builds and returns Dataloader."""

	transform = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	dataset = ImageFolder(mode, image_path, image_size, transform)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader
