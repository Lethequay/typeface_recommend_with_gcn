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
	def __init__(self, data_path, image_path, transform=None):
		"""Initializes image paths and preprocessing module."""
		self.data_arr = np.load(data_path)
		self.data_size = len(self.data_arr)
		self.image_path = image_path
		self.image_paths = list(map(lambda x: os.path.join(image_path, x), os.listdir(image_path)))
		self.transform = transform
		print("data count :", self.data_size)
		print("image count :", len(self.image_paths))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		tokens = self.data_arr[index]
		typography = tokens[0]
		text = list(map(int, tokens[1:]))
		text = torch.from_numpy(np.asarray(text))

		image = Image.open(self.image_path+
						   typography.replace(' ','_').replace('/','-')+'.png')
		size1 = image.size
		image = image.resize((440, 231), Image.ANTIALIAS)
		size2 = image.size
		if self.transform is not None:
			image = self.transform(image)
		size3 = image.size()

		return typography, image, text

	def __len__(self):
		"""Returns the total number of font files."""
		return self.data_size


def get_loader(data_path, image_path, batch_size, num_workers=2):
	"""Builds and returns Dataloader."""

	transform = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	dataset = ImageFolder(data_path, image_path, transform)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader
