import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def main(config):
	cudnn.benchmark = True

	train_loader = get_loader(mode='train',
							  data_path=config.data_path,
							  image_path=config.img_path,
							  image_size=config.image_size,
							  batch_size=config.batch_size,
							  num_workers=config.num_workers)
	test_loader = get_loader(mode='test',
							 data_path=config.data_path,
							 image_path=config.img_path,
							 image_size=config.image_size,
							 batch_size=config.batch_size,
							 num_workers=config.num_workers)

	solver = Solver(config, train_loader, test_loader)

	# Create directories if not exist
	if not os.path.exists(config.model_path):
		os.makedirs(config.model_path)
	if not os.path.exists(config.sample_path):
		os.makedirs(config.sample_path)
	if not os.path.exists(config.result_path):
		os.makedirs(config.result_path)

	# Train and sample the images
	if config.mode == 'train':
		solver.train()
	elif config.mode == 'sample':
		solver.sample()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# model hyper-parameters
	parser.add_argument('--image_size', default=(256, 32))
	parser.add_argument('--word_dim', type=int, default=300)
	parser.add_argument('--z_dim', type=int, default=300)
	parser.add_argument('--text_maxlen', type=int, default=300)
	parser.add_argument('--num_typo', type=int, default=2349)

	# training hyper-parameters
	parser.add_argument('--sample_epochs', type=int, default=50)
	parser.add_argument('--start_epochs', type=int, default=0)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--num_epochs_decay', type=int, default=50)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--num_workers', type=int, default=8)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--lambda_cls', type=float, default=10)
	parser.add_argument('--beta1', type=float, default=0.5)		# momentum1 in Adam
	parser.add_argument('--beta2', type=float, default=0.999)	  # momentum2 in Adam

	# misc
	parser.add_argument('--mode', type=str, default='train')
	parser.add_argument('--model_path', type=str, default='./models')
	parser.add_argument('--sample_path', type=str, default='./samples')
	parser.add_argument('--result_path', type=str, default='./results')
	parser.add_argument('--log_path', type=str, default='./logs')
	parser.add_argument('--data_path', type=str, default='./data/fiu_indexed.npy')
	parser.add_argument('--img_path', type=str, default='./data/idx_png/')
	parser.add_argument('--log_step', type=int , default=300)
	parser.add_argument('--val_step', type=int , default=1)

	config = parser.parse_args()
	print(config)
	main(config)
