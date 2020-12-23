#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from torchvision import  transforms
import numpy as np
import random
import sys
from torchvision import datasets as dataset
import pickle
import os
from sklearn import metrics

dataroot = '/tmp'
from data.single_dataset import SingleDataset

dataset_path = 'cov_local_data.pkl'
test_path = 'cov_test_data.pkl'
svhn_size = 3000
mnist_size = 1000



def load_digit():
    idx = 0
    count = 0
    local_img = []
    local_label = []
    dic_users = {}
    
    """
    dataset_m = dataset.MNIST('./data/mnist/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    """
    dataset_m = dataset.MNIST('./data/mnist/', train = True, download=True)
    dataset_s = dataset.SVHN(root =dataroot, split ='extra', download = True)
    data_m = [data[0].convert(mode='RGB') for idx, data in enumerate(dataset_m)]
    print('mnist data shape', data_m[0], dataset_m[0][0].size)
    data_s = [data[0].convert(mode='RGB') for idx, data in enumerate(dataset_s)]
    print('svhn data shape', data_s[0].size, dataset_s[0][0].size)
    label_s = np.array(dataset_s.labels)
    label_m = np.asarray(dataset_m.targets)
    m_num_agent = 60
    s_num_agent = 140
    count = 0
    idx = 0
    
    
    for i in range(m_num_agent):
        start = i * mnist_size
        end = (i + 1) * mnist_size
        for j in range(start, end):
            local_img.append(data_m[j])
            local_label.append(label_m[j])
        dic_users[i+idx] = list(range(count + start, count+end))
    idx+=m_num_agent
    count+=end
    
    for i in range(s_num_agent):
        start = i*svhn_size
        end = (i +1)*svhn_size
        dic_users[i+idx] = list(range(count+start, count+end))
        for j in range(start, end):
            local_img.append(data_s[j])
            local_label.append(label_s[j])
    count+=end
    idx+=s_num_agent

    print('total number of agents', idx)
    return dic_users,local_img, local_label
        





class Custom_Dataset(Dataset):
    def __init__(self, dataset, transform = None, **kwargs):
        self.dataset = dataset
        self.transform = transform


        if os.path.exists(dataset_path):
            with open(dataset_path,'rb') as f:
                save_dataset = pickle.load(f)
                self.img_set = save_dataset['images']
                self.label_set = save_dataset['labels']
                self.dic_user = save_dataset['idx']
        else:
            dic_user, img_set, label_set = load_digit()
            save_dataset = {}
            save_dataset['images'] = img_set
            save_dataset['labels'] = label_set
            save_dataset['idx'] = dic_user
            with open(dataset_path, 'wb') as f:
                pickle.dump(save_dataset, f)
            self.img_set = img_set
            self.label_set = label_set
            self.dic_user = dic_user



    def __len__(self):
        return len(self.label_set)

    def __getitem__(self, item):
        image = self.img_set[item]
        label = self.label_set[item]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

class Server_Dataset(Dataset):
    def __init__(self, dataset, transform = None,da=False, **kwargs):
        self.dataset = dataset
        self.transform = transform
        if da == True:
            save_path = 'da'+test_path
        else:
            save_path = test_path
        if os.path.exists(save_path):
            with open(save_path,'rb') as f:
                save_dataset = pickle.load(f)
                self.img_set = save_dataset['images']
                self.label_set = save_dataset['labels']
        else:
            img_set, label_set = prepare_server(da=da)
            save_dataset = {}
            save_dataset['images'] = img_set
            save_dataset['labels'] = label_set
            with open(save_path, 'wb') as f:
                pickle.dump(save_dataset, f)
            self.img_set = img_set
            self.label_set = label_set



    def __len__(self):
        return len(self.label_set)

    def __getitem__(self, item):
        image = self.img_set[item]
        label = self.label_set[item]
        if self.transform is not None:
            image = self.transform(image)
        return image, label




def prepare_server(da=False):
    """
    Prepare data for server, assumping the uniform distribution
    pick 10% data from all clients
    """

    ratio = 1 #take 10% as the test global data
    if da:
        ratio = 1
        dataset_u = dataset.USPS(root=dataroot, train=True, download=True)
    else:
        dataset_u = dataset.USPS(root=dataroot, train=False, download=True)
    data_u = [data[0].resize((32,32)).convert(mode='RGB') for idx, data in enumerate(dataset_u)]
    # 64 is used for alexnet
    label_u = np.asarray(dataset_u.targets)

    global_data = []
    global_label = []

    index = np.array(np.arange(len(label_u)))
    #np.random.shuffle(index)
    keep_len = int(len(index)* ratio)
    index = index[:keep_len]

    for i in index:
        global_data.append(data_u[i])
        global_label.append(label_u[i])

    print('size of test data', len(global_label))
    return global_data, global_label
