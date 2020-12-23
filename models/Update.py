#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import torch.nn.functional as F
from sklearn import metrics
import sys
sys.path.append('../')

from models.Nets import Discriminator
def custom_scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
def entropy_loss(v):
        """
          Entropy loss for probabilistic prediction vectors
          input: batch_size x channels x h x w
          output: batch_size x 1 x h x 2
        """
        dim =  v.dim()
        num = 1
        for i in range(dim):
                num = num * v.shape[i]
        _loss = -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (num+1e-30)
        return _loss


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, iter = None):
        self.args = args
        if args.dataset == 'face':
            self.loss_func = nn.MultiLabelSoftMarginLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        if iter is not None:
            self.iter = iter
        else:
            self.iter = 0
    def train(self, net):
        net.train()
        # Comment the following if freeze gradients
        """    
        for layer_idx, param in enumerate(net.feature.parameters()):
            param.requires_grad = False
    
        for layer_idx, param in enumerate(net.parameters()):
            param.requires_grad = False
        for layer_idx, param in enumerate(net.classifier.parameters()):
            param.requires_grad = True
        """
        # train and update
        lr = self.iter

        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)


        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if batch_idx > self.args.local_it:
                    #run local_it before stop
                    break
                labels = torch.tensor(labels, dtype =torch.long)
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
                #if batch_idx %40 ==0:
                #    print('cur batch', batch_idx, '/', len(self.ldr_train), 'loss', loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)



class DA_LocalUpdate(object):
    def __init__(self, args, dataset=None,tgt_dataset = None, idxs=None, iter = None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_server = DataLoader(tgt_dataset, batch_size=self.args.local_bs, shuffle=True)
        if iter is not None:
            self.iter = iter
        else:
            self.iter = 0
    def train(self, net):
        discriminator = Discriminator(feat_dim =384, num_domain = 2).to(self.args.device)
        criterion =torch.nn.CrossEntropyLoss()
        net.train()
        discriminator.train()
        # train and update
        lr = self.iter
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)
        optimizer_d = torch.optim.SGD(discriminator.parameters(), lr=lr, momentum=0.5)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, ((src_images, src_labels), (tgt_images, tgt_labels)) in enumerate(zip((self.ldr_train), self.ldr_server)):
                if batch_idx > self.args.local_it:
                    #run local_it before stop
                    break
                src_images, src_labels, tgt_images, tgt_labels= src_images.to(self.args.device), src_labels.to(self.args.device), tgt_images.to(self.args.device), tgt_labels.to(self.args.device)

                src_feat, src_outputs = net(src_images)
                tgt_feat, tgt_outputs = net(tgt_images)
                soft_tgt_outputs = F.softmax(tgt_outputs, dim=0) # new ad
                concat_feat = torch.cat((src_feat, tgt_feat),0)
                d_output = discriminator(concat_feat)
                domain_label = torch.cat((torch.ones(src_images.shape[0]), torch.zeros(tgt_images.shape[0])), 0)
                domain_label = domain_label.long().cuda()
                loss_d = F.nll_loss(d_output, domain_label)
                net.requires_grad = False
                discriminator.requires_grad = True
                optimizer_d.zero_grad()
                loss_d.backward(retain_graph = True)
                optimizer_d.step()

                #adversarial domain adaptation
                d_tgt_output = discriminator(tgt_feat)
                tgt_guide = torch.ones(tgt_images.shape[0]).long().cuda()
                loss_adv = F.nll_loss(d_tgt_output, tgt_guide)
                loss_recog = criterion(src_outputs, src_labels.long())
                loss_tgt = entropy_loss(soft_tgt_outputs)
                loss_model = loss_recog + loss_adv  + loss_tgt
                net.requires_grad = True
                discriminator.requires_grad = False
                optimizer.zero_grad()
                loss_model.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss_model.item()))
                batch_loss.append(loss_model.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

