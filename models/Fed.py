#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


resnet = False
def noisy_FedAvg(w, global_model, idx=None):
    w_avg = copy.deepcopy(global_model)
    coeff_list = []
    updates = [{} for i in range(len(w))]
    noise_multiplier = 0.06
    l2_norm_clip = 0.08
    #print('global model', global_model.data)
    #print(' local weight shape', w[0])
    avg_norm = 0
    print('length of parameters', len(w))
    for i in range(0, len(w)):
        cur_norm = 0
        for idx,k in enumerate(w_avg.keys()):
            #if idx <len(w_avg.keys())-2:
            #   continue
            #print('remain k',k)
            if  w_avg[k].dtype !=torch.float:# or resnet:
                continue
            updates[i][k] = w[i][k] - global_model[k]
            cur_norm+=updates[i][k].norm(2).item()**2
        cur_norm = cur_norm ** .5
        avg_norm+=cur_norm
        #clip_coeff = 1.0
        #if cur_norm > l2_norm_clip:
        #    print('cur_norm', cur_norm)
        clip_coeff = min(l2_norm_clip / (cur_norm + 1e-6), 1.)
        coeff_list.append(clip_coeff)

    #print('avg norm', cur_norm/len(w))
    
    for i in range(0, len(w)):
        for idx,k in enumerate(w_avg.keys()):
            #if idx<len(w_avg.keys())-2:
            #    continue
            if w_avg[k].dtype !=torch.float:
                continue
            updates[i][k] = updates[i][k] * coeff_list[i]
    
    for idx,k in enumerate(w_avg.keys()):
       
        if  w_avg[k].dtype !=torch.float:
                continue
        #if idx<len(w_avg.keys())-2:
        #    continue
        w_avg[k] = len(w)*w_avg[k]
        for i in range(0, len(w)):
            w_avg[k]+=updates[i][k]
        #print('key', k, 'noisy scale', 1.0/len(w)*noise_multiplier*torch.randn_like(w_avg[k].data))
        w_avg[k] = torch.div(w_avg[k], len(w)) + 1.0/len(w)*noise_multiplier*torch.randn_like(w_avg[k].data)

    
    
    #print('coeff_list', coeff_list)
    """
        total_norm = 0
        for k in w_avg.keys():
            total_norm+=w[i][k].norm(2).item() ** 2


    for k in w_avg.keys():
        total_norm+=
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    """
    return w_avg
