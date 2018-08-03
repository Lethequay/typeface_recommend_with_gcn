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
	def __init__(self, mode, data_path, image_path, image_size, transform=None):
		"""Initializes image paths and preprocessing module."""
		self.mode = mode
		self.data_arr = np.load(data_path)
		shuffle(self.data_arr)

		self.idx2text = np.load('../data/idx2text.npy')
		self.idx2typos = np.load('../data/idx2typos.npy')
		self.typo_list = np.load('../data/typo_list.npy')
		self.typo_cnt = len(self.typo_list)

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
		print("image count :", len(self.image_paths))
		print("data count :", self.data_size)

	def preprocess(self):
		train_thr = int(len(self.data_arr) * 0.8)
		for i, data in enumerate(self.data_arr):
			if (i+1) < train_thr:
				self.train_dataset.append(data)
			else:
				self.test_dataset.append(data)

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		data = self.train_dataset if self.mode == 'train' else self.test_dataset
		tokens = data[index]
		idx = int(tokens[0])
		typography = tokens[1]
		typos= self.idx2typos[idx]
		text = self.idx2text[idx]
		length = int(text[-1])
		text = text[:-1]
		text = torch.from_numpy(np.asarray(text))

		pos_image = Image.open(self.image_path+str(typography)+'.png').convert("L")
		neg_image = Image.open(self.image_path+str(random.choice([x for x in range(self.typo_cnt)
										  if x not in typos]))+'.png').convert("L")
		pos_image = pos_image.resize(self.image_size, Image.ANTIALIAS)
		neg_image = neg_image.resize(self.image_size, Image.ANTIALIAS)
		if self.transform is not None:
			pos_image = self.transform(pos_image)
			neg_image = self.transform(neg_image)
		text_typo_label = torch.from_numpy(np.asarray(
						  [1 if i in typos else 0 for i in range(self.typo_cnt)]))

		return typography, pos_image, neg_image, text, length, text_typo_label

	def __len__(self):
		"""Returns the total number of font files."""
		return self.data_size


def get_loader(mode, data_path, image_path, image_size, batch_size, num_workers=2):
	"""Builds and returns Dataloader."""

	transform = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	dataset = ImageFolder(mode, data_path, image_path, image_size, transform)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader
