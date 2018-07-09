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

		self.typo_dict = np.load('./data/typo_dict.npy').item()
		self.typo_size = len(self.typo_dict)

		self.image_size = image_size
		self.image_path = image_path
		self.image_paths = list(map(lambda x: os.path.join(image_path, x), os.listdir(image_path)))
		self.dataidx2typos = np.load('./data/idx2typos.npy').item()

		self.transform = transform
		self.train_dataset = []
		self.test_dataset = []
		self.preprocess()

		if mode == 'train':
			self.data_size = len(self.train_dataset)
		else:
			self.data_size = len(self.test_dataset)
		print("typo count :", self.typo_size)
		print("data count :", self.data_size)
		print("image count :", len(self.image_paths))

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
		text = list(map(int, tokens[2:-1]))
		text = torch.from_numpy(np.asarray(text))
		length = int(tokens[-1])

		pos_image = Image.open(self.image_path+
							   typography.replace(' ','_').replace('/','=')+'.png')\
							   .convert("L")
		neg_image = Image.open(random.choice([x for x in self.image_paths
							   if x[len(self.image_path):-len('.png')] not in self.dataidx2typos[idx]]))\
							   .convert("L")
		pos_image = pos_image.resize(self.image_size, Image.ANTIALIAS)
		neg_image = neg_image.resize(self.image_size, Image.ANTIALIAS)
		if self.transform is not None:
			pos_image = self.transform(pos_image)
			neg_image = self.transform(neg_image)
		typography = self.typo_dict[typography]

		return typography, pos_image, neg_image, text, length

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
