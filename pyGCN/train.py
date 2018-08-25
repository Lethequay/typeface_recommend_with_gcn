from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import tensorboardX
import os
import glob
from torch.autograd import Variable

from utils import *
from model import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=30, help='Patience')

# model hyper-parameters
parser.add_argument('--image_size', default=(256, 32))
parser.add_argument('--word_dim', type=int, default=300)
parser.add_argument('--z_dim', type=int, default=300)
parser.add_argument('--text_maxlen', type=int, default=300)
parser.add_argument('--num_typo', type=int, default=2349)
parser.add_argument('--at', type=int, default=30)

# training hyper-parameters
parser.add_argument('--sample_epochs', type=int, default=0)
parser.add_argument('--start_epochs', type=int, default=0)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--num_epochs_decay', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lambda_cls', type=float, default=10)
parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')

# misc
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--model_path', type=str, default='./models')
parser.add_argument('--sample_path', type=str, default='./samples')
parser.add_argument('--result_path', type=str, default='./results')
parser.add_argument('--log_path', type=str, default='./logs')
parser.add_argument('--data_path', type=str, default='../data/fiu_indexed.npy')
parser.add_argument('--img_path', type=str, default='../data/idx_png/')
parser.add_argument('--adj_path', type=str, default='../data/matrix/text_typo_mat.npy')
parser.add_argument('--typo_path', type=str, default='../data/typo_list.npy')
parser.add_argument('--log_step', type=int , default=300)
parser.add_argument('--val_step', type=int , default=5)

text_cnt = 6237#text_feat.size(0)
img_cnt  = 2349#img_feat.size(0)
text_train = range(int(text_cnt*0.7))
text_val   = range(int(text_cnt*0.7), int(text_cnt*0.8))
text_test  = range(int(text_cnt*0.8), text_cnt)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device('cuda' if args.cuda else 'cpu')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
text_loader_ = text_loader(args.batch_size)
image_loader = img_loader(args.img_path, args.image_size, args.batch_size)
#pair_loader = pair_loader(args.data_path, args.batch_size)

# Adj. Matrix
adj = torch.from_numpy(np.load(args.adj_path)).type(torch.FloatTensor)
adj[text_val] = 0
adj[:,text_val]= 0
adj[text_test] = 0
adj[:,text_test]= 0
adj = normalize(adj + torch.eye(adj.size(0)))

# Model and optimizer
joint_model = GCN(nfeat=args.z_dim, nhid=args.z_dim//2, nclass=int(args.num_typo), dropout=args.dropout)
text_model = Text_Encoder(args.word_dim, args.z_dim, args.text_maxlen)
image_model = Resnet(args.z_dim, args.num_typo, args.image_size)
image_opt = optim.Adam(image_model.parameters(), args.lr, [args.beta1, args.beta2], weight_decay=args.weight_decay)
optimizer = optim.Adam(list(text_model.parameters()) + list(image_model.parameters())\
                       + list(joint_model.parameters()),
                       args.lr, [args.beta1, args.beta2], weight_decay=args.weight_decay)

if args.cuda:
    joint_model.cuda()
    text_model.cuda()
    image_model.cuda()
    adj = Variable(adj, requires_grad=False).type(torch.cuda.FloatTensor)
else:
    adj = Variable(adj, requires_grad=False).type(torch.FloatTensor)

#print_network(joint_model, 'JM')
#print_network(text_model, 'TM')
#print_network(image_model, 'IM')

def pretrain(epoch):

    acc_img = 0
    # Image Embedding
    for i, (label, image) in enumerate(image_loader):
        label = label.to(args.device)
        image = image.to(args.device)
        image_model.train()

        _, img_cls = image_model(image)
        loss_img = F.cross_entropy(img_cls, label)
        acc_img += accuracy_at_k(img_cls, label)

        image_opt.zero_grad()
        loss_img.backward()
        image_opt.step()

    print("[Epoch #{}] Img Cls Acc@5: {}".format(epoch, acc_img/(i+1)))


writer = tensorboardX.SummaryWriter(args.log_path)
def train(epoch):

    t = time.time()
    joint_model.train()
    text_model.train()
    image_model.train()

    # Text Embedding
    text_feat = []
    text_typo_list = []
    for i, text, text_len, text_typo_label  in text_loader_:
        text = text.to(args.device)
        text_len = text_len.to(args.device)
        text_typo_label = text_typo_label.to(args.device)

        text_emb, text_cls = text_model(text, text_len)
        text_feat.append(text_emb)
        text_typo_list.append(text_typo_label)
    text_feat = torch.cat(text_feat, 0)
    text_typo_list = torch.cat(text_typo_list, 0).type(torch.FloatTensor)

    # Image Embedding
    img_feat = []
    img_label = []
    for label, image in image_loader:
        label = label.to(args.device)
        image = image.to(args.device)

        style_emb, style_cls = image_model(image)
        img_feat.append(style_emb)
        img_label.append(label)
    img_feat = torch.cat(img_feat, 0)
    img_label = torch.cat(img_label, 0)

    # Joint Embedding
    output, text_cls, img_cls = joint_model(torch.cat((text_feat, img_feat), 0), adj)

    # Loss & Acc.
    loss_text= F.binary_cross_entropy_with_logits(text_cls[text_train], text_typo_list[text_train])
    acc_text = baccuracy_at_k(text_cls[text_train], text_typo_list[text_train])
    loss_img = F.cross_entropy(img_cls, img_label)
    acc_img  = accuracy_at_k(img_cls, img_label)

    loss_train = loss_text + loss_img

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    joint_model.eval()
    text_model.eval()
    image_model.eval()
    with torch.no_grad():
        # Vector Similarity
        vec_sim = torch.mm(output[text_val], output[range(text_cnt,text_cnt+img_cnt)].t())
        _, sort_idx = torch.sort(vec_sim, 1, descending=True)
        mat_cnt = sum([j in sort_idx[i,:30] for (i,j) in torch.nonzero(text_typo_list[text_val])])
        precision = mat_cnt/(30*len(text_val))
        recall = mat_cnt/len(torch.nonzero(text_typo_list[text_val]))
        print("Sim Prec@30: {:.4f} ".format(precision), end='')
        print("Sim Recall@30: {:.4f} ".format(recall), end='')
        writer.add_scalar('Sim Prec@30', precision, epoch)
        writer.add_scalar('Sim Recall@30', recall, epoch)

        loss_val = F.binary_cross_entropy_with_logits(text_cls[text_val], text_typo_list[text_val])
        acc_text_val = baccuracy_at_k(text_cls[text_val], text_typo_list[text_val])
        writer.add_scalar('Text Cls Acc@30', acc_text_val, epoch)
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_text: {:.4f}'.format(loss_text.item()),
              'loss_img: {:.4f}'.format(loss_img.item()),
              'text_acc_train: {:.4f}'.format(acc_text),
              'img_acc_train: {:.4f}'.format(acc_img),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'text_acc_val: {:.4f}'.format(acc_text_val),
              'time: {:.4f}s'.format(time.time() - t))

    return recall


def compute_test():
    joint_model.eval()
    text_model.eval()
    image_model.eval()

    # Text Embedding
    text_feat = []
    text_typo_list = []
    for i, text, text_len, text_typo_label  in text_loader_:
        text = text.to(args.device)
        text_len = text_len.to(args.device)
        text_typo_label = text_typo_label.to(args.device)

        text_emb, text_cls = text_model(text, text_len)
        text_feat.append(text_emb)
        text_typo_list.append(text_typo_label)
    text_feat = torch.cat(text_feat, 0)
    text_typo_list = torch.cat(text_typo_list, 0).type(torch.FloatTensor)

    # Image Embedding
    img_feat = []
    img_label = []
    for label, image in image_loader:
        label = label.to(args.device)
        image = image.to(args.device)

        style_emb, style_cls = image_model(image)
        img_feat.append(style_emb)
        img_label.append(label)
    img_feat = torch.cat(img_feat, 0)
    img_label = torch.cat(img_label, 0)

    # Joint Embedding
    output, text_cls, img_cls = joint_model(torch.cat((text_feat, img_feat), 0), adj)

    # Vector Similarity
    vec_sim = torch.mm(output[text_test], output[range(text_cnt,text_cnt+img_cnt)].t())
    _, sort_idx = torch.sort(vec_sim, 1, descending=True)
    mat_cnt = sum([j in sort_idx[i,:args.at] for (i,j) in torch.nonzero(text_typo_list[text_test])])
    print("Sim Prec@30: {:.4f} ".format(mat_cnt/(args.at*len(text_test))), end='')
    print("Sim Recall@30: {:.4f} ".format(mat_cnt/len(torch.nonzero(text_typo_list[text_test]))), end='')
    print("mat_cnt :",mat_cnt, ", 30*text_test :", args.at*len(text_test), ", len(nonzero) :", len(torch.nonzero(text_typo_list[text_test])))

    loss_val = F.binary_cross_entropy_with_logits(text_cls[text_test], text_typo_list[text_test])
    acc_text_val = baccuracy_at_k(text_cls[text_test], text_typo_list[text_test], k=args.at)
    print('Val Loss: {:.4f}, Text Acc@30: {:.4f}'.format(loss_val/len(text_test), acc_text_val))

    # Vector Visualization
    text_label= list(map(str, range(text_cnt)))
    typo_label = list(map(str, np.load(args.typo_path)))
    writer.add_embedding(output, text_label+typo_label)


if args.sample_epochs:
    print('Loading {}th epoch'.format(args.sample_epochs))
    joint_model.load_state_dict(torch.load(args.model_path+'/jm-{}.pkl'.format(args.sample_epochs)))
    text_model.load_state_dict(torch.load(args.model_path+'/tm-{}.pkl'.format(args.sample_epochs)))
    image_model.load_state_dict(torch.load(args.model_path+'/im-{}.pkl'.format(args.sample_epochs)))

# Train model
t_total = time.time()
bad_counter = 0
best = 0
best_epoch = 0

if args.mode == 'train':
    print('Pretraining the image model...')
    for epoch in range(30):
        pretrain(epoch)
    torch.save(image_model.state_dict(), args.model_path+'/preim-{}.pkl'.format(30))
    image_model.load_state_dict(torch.load(args.model_path+'/preim-{}.pkl'.format(30)))
    print('Main Train Start')
    for epoch in range(args.sample_epochs, args.epochs):
        train(epoch)

        if epoch > 70:
            torch.save(joint_model.state_dict(), args.model_path+'/jm-{}.pkl'.format(epoch))
            torch.save(text_model.state_dict(), args.model_path+'/tm-{}.pkl'.format(epoch))
            torch.save(image_model.state_dict(), args.model_path+'/im-{}.pkl'.format(epoch))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
elif args.mode == 'test':
    best_epoch = args.sample_epochs

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
joint_model.load_state_dict(torch.load(args.model_path+'/jm-{}.pkl'.format(best_epoch)))
text_model.load_state_dict(torch.load(args.model_path+'/tm-{}.pkl'.format(best_epoch)))
image_model.load_state_dict(torch.load(args.model_path+'/im-{}.pkl'.format(best_epoch)))

# Testing
compute_test()
