from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
import os
import glob
from torch.autograd import Variable

from utils import *
from models import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

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
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--lambda_cls', type=float, default=10)
parser.add_argument('--beta1', type=float, default=0.5)		# momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)	  # momentum2 in Adam
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')

# misc
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--model_path', type=str, default='./models')
parser.add_argument('--sample_path', type=str, default='./samples')
parser.add_argument('--result_path', type=str, default='./results')
parser.add_argument('--data_path', type=str, default='../data/fiu_indexed.npy')
parser.add_argument('--img_path', type=str, default='../data/idx_png/')
parser.add_argument('--adj_path', type=str, default='../data/matrix/typo2typo_mat.npy')
parser.add_argument('--log_step', type=int , default=300)
parser.add_argument('--val_step', type=int , default=5)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj = torch.from_numpy(np.load(args.adj_path)).type(torch.FloatTensor)
data_loader = get_loader(args.data_path, args.img_path, args.image_size, args.batch_size)
idx_train = range(int(len(data_loader)*0.8))
idx_val   = range(int(len(data_loader)*0.8), len(data_loader))

# Model and optimizer
joint_model = GCN(nfeat=args.z_dim, nhid=args.z_dim//2, nclass=int(args.num_typo), dropout=args.dropout)
#text_encoder = Text_Encoder(args.word_dim, args.z_dim, args.text_maxlen)
image_model = Resnet(args.z_dim, args.num_typo, args.image_size)
#optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, text_encoder.parameters())) + \
optimizer = optim.Adam(list(joint_model.parameters()) + list(image_model.parameters()),
                       args.lr, [args.beta1, args.beta2], weight_decay=args.weight_decay)

if args.cuda:
    joint_model.cuda()
    image_model.cuda()
adj = normalize(adj + torch.eye(adj.shape[0]))
adj = Variable(adj).type(torch.cuda.FloatTensor)

print_network(joint_model, 'JM')
print_network(image_model, 'IM')


def train(epoch):

    t = time.time()
    joint_model.train()
    image_model.train()
    optimizer.zero_grad()

    # Image Embedding
    features = []
    labels = []
    for i, (label, image) in enumerate(data_loader):
        image = Variable(image).cuda()
        label = Variable(label).cuda()

        _, style_emb, _ = image_model(image)
        features.append(style_emb)
        labels.append(label)
    features = torch.cat(features, 0)
    labels = torch.cat(labels, 0)

    output = joint_model(features, adj)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy_at_k(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        joint_model.eval()
        output = joint_model(features, adj)

    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = accuracy_at_k(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc@5_val: {:.4f}'.format(acc_val),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.item()


def compute_test():
    joint_model.eval()
    image_model.eval()
    output = model(features, adj)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy_at_k(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test.data[0]))


# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    if loss_values[-1] < best:
        torch.save(joint_model.state_dict(), args.model_path+'/jm-{}.pkl'.format(epoch))
        torch.save(image_model.state_dict(), args.model_path+'/im-{}.pkl'.format(epoch))

        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0

        files = glob.glob(args.model_path+'*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0][len('jm-'):])
            if epoch_nb < best_epoch:
                os.remove(file)
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break


print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
joint_model.load_state_dict(torch.load(args.models_paht+'/jm-{}.pkl'.format(best_epoch)))
image_model.load_state_dict(torch.load(args.models_paht+'/im-{}.pkl'.format(best_epoch)))

# Testing
compute_test()
