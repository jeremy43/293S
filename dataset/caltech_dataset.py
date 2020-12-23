#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import sys
import pickle
import os
from sklearn import metrics

dataroot = '/home/yq/office_caltech_10'
from data.single_dataset import SingleDataset

dataset_path = 'caltech_local_data.pkl'
test_path = 'caltech_test_data.pkl'
size = 256


def load_office():
    with open(os.path.join(dataroot, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    print('length of class', len(classes), classes)
    assert(len(classes) == 10)
    idx = 0
    count = 0
    local_img = []
    local_label = []
    dic_users = {}
    for name in ['amazon', 'dslr', 'caltech']:
        dataset_path = os.path.join(dataroot, name)
        dataset_type = 'SingleDataset'
        dataset = SingleDataset()
        dataset.initialize(root=dataset_path, classnames=classes)
        labels = dataset.data_labels
        data = dataset.data_paths
        index = np.array(np.arange(len(labels)))
        np.random.shuffle(index) #No shuffling, or the cache result is not correct
        data =[data[i] for i in index]
        labels = [labels[i] for i in index]
        print('domain name', name, 'length of data in domain', len(dataset.data_labels))
        data = [da.resize((300,300)).convert(mode='RGB') for idx, da in enumerate(data)]
        # we total have two agents
        num_agent = 1
        size = len(data)
        #size = int(len(data)/2)
        #num_agent = int(len(data)/size)

        for i in range(num_agent):
            start = i * size
            end = (i + 1) * size
            #t_data = data[start : end]
            #t_labels = labels[start: end]
            if i == num_agent -1:
                end = len(data)-1
            dic_users[i+idx] = list(range(count+start, count+end))
            for j in range(start, end):
                local_img.append(data[j])
                local_label.append(labels[j])
        idx+=num_agent
        count+=end
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
            dic_user, img_set, label_set = load_office()
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
    def __init__(self, dataset, transform = None, **kwargs):
        self.dataset = dataset
        self.transform = transform


        if os.path.exists(test_path):
            with open(test_path,'rb') as f:
                save_dataset = pickle.load(f)
                self.img_set = save_dataset['images']
                self.label_set = save_dataset['labels']
        else:
            img_set, label_set = prepare_server()
            save_dataset = {}
            save_dataset['images'] = img_set
            save_dataset['labels'] = label_set
            with open(test_path, 'wb') as f:
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
    ratio = 1.0#take 10% as the global data
    if da == True:
        ratio = 1.0
    global_data = []
    global_label = []
    for name in [ 'webcam']:
        dataset_path = os.path.join(dataroot, name)
        with open(os.path.join(dataroot, 'category.txt'), 'r') as f:
            classes = f.readlines()
            classes = [c.strip() for c in classes]
        assert(len(classes) == 10)
        dataset_type = 'SingleDataset'
        dataset = SingleDataset()
        dataset.initialize(root=dataset_path, classnames=classes)
        labels = dataset.data_labels
        data = dataset.data_paths
        index = np.array(np.arange(len(labels)))
        np.random.shuffle(index) #No shuffling, or the cache result is not correct
        keep_len = int(len(index) * ratio)
        index = index[:keep_len]
        for i in index:
            global_data.append(data[i])
            global_label.append(labels[i])

    data = [da.resize((300,300)).convert(mode='RGB') for idx, da in enumerate(global_data)]
    return data, global_label
