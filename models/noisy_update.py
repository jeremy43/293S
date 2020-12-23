#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from torch.utils.data import TensorDataset
from sklearn import metrics
import sys
sys.path.append('..')
from autodp import rdp_bank, dp_acct, rdp_acct, privacy_calibrator
sys.path.append('./pyvacy')
from pyvacy import optim, analysis, sampling
acct = rdp_acct.anaRDPacct()
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
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        if iter is not None:
            self.iter = iter
        else:
            self.iter = 0
    def train(self, net):
        net.train()
        # train and update
        noise_multiplier =  0.35#0.5
        micro_batch = self.args.local_bs
        optimizer = optim.DPSGD(
        l2_norm_clip=0.08,#orignial set 0.08
        noise_multiplier=noise_multiplier,
        minibatch_size= self.args.local_bs,
        microbatch_size=micro_batch,
        params=net.parameters(),
        lr= self.args.lr,
        momentum = 0.5,
    )


        #optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        


        num_iteration = max(1,self.args.local_it)
        minibatch_loader, microbatch_loader = sampling.get_data_loaders(
        self.args.local_bs,
        micro_batch,
        num_iteration)
        gaussian = lambda x: rdp_bank.RDP_gaussian({'sigma': noise_multiplier*np.sqrt(3)}, x)
        acct = rdp_acct.anaRDPacct()
        #print('num of iteration', num_iteration)
        q = 1.0/len(self.ldr_train)
        acct.compose_poisson_subsampled_mechanisms(gaussian, q, coeff =180* num_iteration)
        print('total iteration', 130*num_iteration, 'sample ratio',q,'auto dp privacy cost', acct.get_eps(1e-4))
        #print('sample ratio',1.0/len(self.ldr_train), 'size of loader', len(self.ldr_train),'batch_size', micro_batch)
        """
        print('Achieves ({}, {})-DP'.format(
        analysis.epsilon(
            len(self.ldr_train)*self.args.local_bs,
            self.args.local_bs,
            noise_multiplier,
            num_iteration,
            1e-4
        ),
        1e-4,
    ))
        """
        iteration = 0
        epoch_loss = []
        for iter in range(1):
        #for iter in range(self.args.local_ep): 
            for x_minibatch, y_minibatch in minibatch_loader(self.ldr_train.dataset):
                optimizer.zero_grad()
                iteration+=1
                if iteration > self.args.local_it:
                    break
                batch_loss = []
                for images, labels in microbatch_loader(TensorDataset(x_minibatch, y_minibatch)):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    optimizer.zero_microbatch_grad()
                    #net.zero_grad()
                    log_probs = net(images)
                    #print('log_probs shape', log_probs.shape)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.microbatch_step()
                    batch_loss.append(loss.item())
                optimizer.step()

            #if iter %8 == 0:
            #    print('epoch', iter,loss.item())
            #print('Update Epoch:  [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #             iteration * len(images), len(self.ldr_train.dataset),
            #                    loss.item()))
            
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            #optimizer.step()
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

