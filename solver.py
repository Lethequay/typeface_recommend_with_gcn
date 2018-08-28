import os
import numpy as np
import time
import datetime
import torch
import torchvision
import tensorboardX
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from data_loader import img_loader
from model import *

class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.image_loader = img_loader(config.img_path, config.image_size, config.batch_size)

        # Models
        self.text_encoder = None
        self.image_encoder = None
        self.distance = None

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
        self.at = 30

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
        self.writer = tensorboardX.SummaryWriter(config.log_path)

    def build_model(self):
        self.text_encoder = Text_Encoder(self.word_dim, self.z_dim, self.text_maxlen)
        self.image_encoder = Resnet(self.z_dim, self.num_typo, self.image_size)
        #self.distance = Mahalanobis_dist(self.z_dim)
        self.distance = Angular_loss()
        self.optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, self.text_encoder.parameters())) + \
                                    list(self.image_encoder.parameters()) + list(self.distance.parameters()),
                                    self.lr, [self.beta1, self.beta2])

        if torch.cuda.is_available():
            self.text_encoder.cuda()
            self.image_encoder.cuda()
            self.distance.cuda()
        #self.print_network(self.text_encoder, 'TE')
        #self.print_network(self.image_encoder, 'IE')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))


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
            print('te/ie is Successfully Loaded from %d epoch'%resume_iters)
            return resume_iters
        else:
            return 0

    def sim_score(self, text_emb, text_typo_label):
        with torch.no_grad():
            features = []
            for i, (label, image) in enumerate(self.image_loader):
                image = Variable(image).cuda()
                style_emb, _ = self.image_encoder(image)
                features.append(style_emb)
            features = torch.cat(features, 0)

            text_sim = torch.mm(text_emb, features.t())
            _, sort_idx = torch.sort(text_sim)
            mat_cnt = sum([j in sort_idx[i,:self.at] for (i,j) in torch.nonzero(text_typo_label)])
            precision = mat_cnt/(self.at*text_emb.size(0))
            recall = mat_cnt/len(torch.nonzero(text_typo_label))

            print("mat_cnt :",mat_cnt, ", 30*text_emb :", self.at*text_emb.size(0), ", len(nonzero) :", len(torch.nonzero(text_typo_label)))
        return precision, recall


    def train(self):

        start_iter = self.restore_model(self.start_epochs)
        iters_per_epoch = len(self.train_loader)
        start_time = time.time()

        for epoch in range(start_iter, self.num_epochs):
            text_acc = 0.
            img_acc  = 0.
            #======================================= Train =======================================#
            for i, (typography, pos_image, neg_image, text, text_len, text_typo_label) in enumerate(self.train_loader):
                loss = {}
                pos_image = pos_image.to(self.device)
                neg_image = neg_image.to(self.device)
                text = text.to(self.device)
                text_len = text_len.to(self.device)
                text_typo_label = text_typo_label.type(torch.cuda.FloatTensor)

                # Text Embedding
                # (batch x z_dim)
                text_emb, text_cls  = self.text_encoder(text, text_len)
                loss_text_cls = F.binary_cross_entropy_with_logits(text_cls, text_typo_label)

                # Image Embedding
                # (batch x z_dim), (batch x z_dim)
                pos_style_emb, out_cls = self.image_encoder(pos_image)
                neg_style_emb, _       = self.image_encoder(neg_image)
                loss_img_cls = F.cross_entropy(out_cls, typography.to(self.device))

                # Joint Embedding
                #loss_joint = F.triplet_margin_loss(text_emb, pos_style_emb, neg_style_emb, margin=1)
                # Mahalanobis distance & Angular_loss
                loss_joint = self.distance(text_emb, pos_style_emb, neg_style_emb)
                # LSE
                #loss_joint = F.mse_loss(text_emb, pos_style_emb, size_average=False)

                # Accuracy
                _, pred  = torch.sort(text_cls, 1, descending=True)
                text_acc+= baccuracy_at_k(pred.data.cpu().numpy(), text_typo_label)
                _, pred  = torch.sort(out_cls, 1, descending=True)
                img_acc += accuracy_at_k(pred.data.cpu().numpy(), typography)

                # Compute gradient penalty
                total_loss = loss_joint + loss_text_cls + loss_img_cls

                # Backprop + Optimize
                self.reset_grad()
                total_loss.backward()
                self.optimizer.step()

                # logging
                loss['text_cls']= loss_text_cls.item()
                loss['img_cls'] = loss_img_cls.item()
                loss['joint']   = loss_joint.item()

                # Print the log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                            elapsed, epoch+1, self.num_epochs, i+1, iters_per_epoch)
                    log += ", text prec@30: {:.4f}".format(text_acc/(i+1))
                    log += ", image prec@5: {:.4f}".format(img_acc/(i+1))

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)


            #==================================== Validation ====================================#
            if (epoch+1) % self.val_step == 0:
                self.text_encoder.eval()
                self.image_encoder.eval()

                val_loss = 0.
                text_acc = 0.
                img_acc  = 0.
                text_feat = []
                text_label= []
                with torch.no_grad():
                    for i, (typography, image, _, text, text_len, text_typo_label) in enumerate(self.valid_loader):
                        image = image.to(self.device)
                        text = text.to(self.device)
                        text_len = text_len.to(self.device)
                        text_typo_label = text_typo_label.type(torch.cuda.FloatTensor)

                        # (batch x z_dim)
                        text_emb, text_cls  = self.text_encoder(text, text_len)
                        text_feat.append(text_emb)
                        text_label.append(text_typo_label)
                        # (batch x z_dim), (batch x z_dim)
                        style_emb, img_cls = self.image_encoder(image)

                        val_loss+= torch.mean((text_emb - style_emb) ** 2).item()

                        _, pred  = torch.sort(text_cls, 1, descending=True)
                        text_acc+= baccuracy_at_k(pred.data.cpu().numpy(), text_typo_label)

                        _, pred  = torch.sort(img_cls, 1, descending=True)
                        img_acc += accuracy_at_k(pred.data.cpu().numpy(), typography)

                text_feat = torch.cat(text_feat, 0)
                text_label= torch.cat(text_label,0)
                precision, recall = self.sim_score(text_feat, text_label)
                print("Val Prec.@30: {:.4f} ".format(precision), end='')
                print("Val Recall@30: {:.4f} ".format(recall), end='')
                self.writer.add_scalar('Val Prec.@30', precision, epoch)
                self.writer.add_scalar('Val Recall.@30', recall, epoch)
                self.writer.add_scalar('Text Cls Acc@30', text_acc/(i+1), epoch)
                print('Val Loss: {:.4f}, Text Acc@30: {:.4f}, Image Acc@5: {:.4f}'\
                      .format(val_loss/(i+1), text_acc/(i+1), img_acc/(i+1)))

                te_path = os.path.join(self.model_path, 'text-encoder-%d.pkl' %(epoch+1))
                ie_path = os.path.join(self.model_path, 'image-encoder-%d.pkl' %(epoch+1))
                torch.save(self.text_encoder.state_dict(), te_path)
                torch.save(self.image_encoder.state_dict(), ie_path)

                self.text_encoder.train(True)
                self.image_encoder.train(True)

    def sample(self):

        self.restore_model(self.sample_epochs)
        self.text_encoder.eval()
        self.image_encoder.eval()

        val_loss = 0.
        text_acc = 0.
        img_acc  = 0.
        text_feat = []
        text_label= []
        with torch.no_grad():
            for i, (typography, image, _, text, text_len, text_typo_label) in enumerate(self.test_loader):
                image = image.to(self.device)
                text = text.to(self.device)
                text_len = text_len.to(self.device)
                text_typo_label = text_typo_label.type(torch.cuda.FloatTensor)

                # (batch x z_dim)
                text_emb, text_cls  = self.text_encoder(text, text_len)
                text_feat.append(text_emb)
                text_label.append(text_typo_label)
                # (batch x z_dim), (batch x z_dim)
                style_emb, img_cls = self.image_encoder(image)

                val_loss+= torch.mean((text_emb - style_emb) ** 2).item()

                _, pred  = torch.sort(text_cls, 1, descending=True)
                text_acc+= baccuracy_at_k(pred.data.cpu().numpy(), text_typo_label)

                _, pred  = torch.sort(img_cls, 1, descending=True)
                img_acc += accuracy_at_k(pred.data.cpu().numpy(), typography)

        text_feat = torch.cat(text_feat, 0)
        text_label= torch.cat(text_label,0)
        precision, recall = self.sim_score(text_feat, text_label)
        print("Val Prec.@30: {:.4f} ".format(precision), end='')
        print("Val Recall@30: {:.4f} ".format(recall), end='')
        print('Val Loss: {:.4f}, Text Acc@30: {:.4f}, Image Acc@5: {:.4f}'\
                      .format(val_loss/(i+1), text_acc/(i+1), img_acc/(i+1)))
