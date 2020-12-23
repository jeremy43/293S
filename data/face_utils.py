from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import os.path as osp
import numpy as np
import torch
import scipy
import math



def Hamming_Score(y_true, y_pred, torch=True,cate=False):
    """
    torch = true mean y_pred is torch tensor
    if torch=false mean y_pred=numpy
    """
    acc_list = []
    if torch:
        from sklearn.metrics import accuracy_score
    for i in range(len(y_true)):
        if torch:
            summary = y_true[i] == y_pred[i].double()
            summary = summary.cpu()
            num = np.sum(summary.numpy())
        else:
            summary = y_true[i] == y_pred[i]
            num = np.sum(summary)
        tmp_a = num / float(len(y_true[i]))
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def hamming_precision(y_true, y_pred,torch = True, cate = True):
    acc_list = []
    if torch:
        from sklearn.metrics import accuracy_score
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
    for i in range(len(y_true)):

        set_true = set( np.where(y_true[i]==1)[0] )
        set_pred = set( np.where(y_pred[i]==1)[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
            float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)
