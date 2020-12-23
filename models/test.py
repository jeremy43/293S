#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
from data.face_utils import Hamming_Score, hamming_precision
import numpy as np

def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    hamming_score = []
    precision =[]
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    criterion = nn.MultiLabelSoftMarginLoss()

    for idx, (data, target) in enumerate(data_loader):
        target = torch.tensor(target, dtype =torch.long)

        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        if args.da:
            feature, log_probs = net_g(data)
        else:
            log_probs = net_g(data)
        # sum up batch loss
        
        # get the index of the max log-probability
        if args.dataset =='face':
            predA = log_probs
            #if idx%100==0:
            #    print('predA', log_probs)
            predAs = torch.round(torch.sigmoid(predA))
            hamming_score.append(Hamming_Score(target, predAs))
            precision.append(hamming_precision(target,predAs))
            test_loss+=criterion(log_probs, target).item()
        else:
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    if args.dataset == 'face':
        mean_hamming_score = np.mean(hamming_score)
        print("mean_hamminng_score: {:.2%}".format(mean_hamming_score))
        test_loss /= len(data_loader.dataset)
        mean_precision = np.mean(precision)
        print('mean precision count one {:.2%}'.format(mean_precision))
        return mean_hamming_score, test_loss
    else:
        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)

        if args.verbose:
            print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
        return accuracy, test_loss
