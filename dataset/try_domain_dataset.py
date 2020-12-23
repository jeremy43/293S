#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as npi
import random
from PIL import Image
import sys
import pickle
import os
from sklearn import metrics
sys.path.append('./')
dataroot = '/home/yq/domain_net'
from domain_data.single_dataset import SingleDataset
from domain_data.manager import custom_dataset
from domain_data.image_folder import make_dataset_with_labels
dataset_path = 'domain_local_data.pkl'
test_path = 'domain_test_data.pkl'
size = 256
import numpy as np


class custom_dataset(object):
    #build a customized dataset

    def __init__(self, root=None,transform=None, **kwargs):
        self.dataset_dir = '/home/yq/domain_net'
        self.image_dir = '/home/yq/domain_net'
        if transform is not None:
            self.transform = transform
        label_path = os.path.join(self.dataset_dir,root+'_train.txt')
        with open(label_path, 'r') as txt:
            train_line = txt.readlines()
        test_path = os.path.join(self.dataset_dir, root +'_test.txt')

        with open(test_path,'r') as txt:
            test_line = txt.readlines()

        train_data, train_label, num_train_imgs = self._process_dir(train_line)
        test_data, test_label, num_test_imgs = self._process_dir(test_line)
        num_total_imgs = num_train_imgs + num_test_imgs
        print(root,"=> dataset loaded")

        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # images")
        print("  ------------------------------")
        print("  train    | {:8d}".format(num_train_imgs))
        print("  test     | {:8d}".format(num_test_imgs))
        print("  ------------------------------")
        print("  total    | {:8d}".format(num_total_imgs))
        print("  ------------------------------")

        self.train_data = train_data
        self.test_data = test_data
        self.train_label = train_label
        self.test_label = test_label
    def _process_dir(self, lines):
        data =[]
        label =[]
        index = np.arange(len(lines))
        np.random.shuffle(index)
        lines = [lines[i] for i in index]
        for img_idx, img_info in enumerate(lines):
            img_path = img_info.split(' ', 1)[0]
            cur_label = img_info.split(' ', 1)[1].split()
            img_path = os.path.join(self.image_dir, img_path)
            if os.path.exists(img_path):
                # save path for each image
                #img = Image.open(img_path).convert('RGB')
                data.append(img_path)
                cur_label = int(cur_label[0])
                label.append(cur_label)
                print('cur_label', cur_label)

        num_imgs = len(data)
        label = np.array(label,dtype=np.int32)
        print('len of label list',len(label))
        return data,label, num_imgs
    def __len__(self):
        return len(self.train_label)

    def __getitem__(self, item):
        path = self.train_data[item]
        image = Image.open(path).convert('RGB')
        label = self.train_label[item]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_office():
    classes = []
    with open(os.path.join(dataroot, 'try_category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]

    print('length of class', len(classes), classes)
    #assert(len(classes) == 10)
    idx = 0
    count = 0
    local_img = []
    local_label = []
    dic_users = {}
    for name in ['clipart','painting','infograph', 'quickdraw', 'sketch']:
        dir_name = os.path.join('/home/yq/domain_net',name)
        path, label = make_dataset_with_labels(dir_name, classes)
        #cur_dataset = custom_dataset(name)
        end = len(label)
        index = np.array(np.arange(end))
        np.random.shuffle(index)
        keep_len = int(len(index)*1.0)
        print('keep length in name', keep_len, name)
        index = index[:keep_len]
        path = [path[i] for i in index]
        label = [label[i] for i in index]
        #data = [da.resize((300,300)).convert(mode='RGB') for idx, da in enumerate(data)]
        for j in range(keep_len):
            local_img.append(path[j])
            local_label.append(label[j])
        dic_users[idx] = list(range( count, count+keep_len))

        idx+=1
        count+=keep_len

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
        path = self.img_set[item]
        label = self.label_set[item]
        image = Image.open(path).convert('RGB')
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
        path = self.img_set[item]
        image = Image.open(path).convert('RGB')
        label = self.label_set[item]
        if self.transform is not None:
            image = self.transform(image)
        return image, label




def prepare_server(da=False):
    """
    Prepare data for server, assumping the uniform distribution
    pick 10% data from all clients
    """
    ratio = 0.5#take 10% as the global data
    if da == True:
        ratio = 0.5
    global_data = []
    global_label = []
    count = 0
    with open(os.path.join(dataroot, 'try_category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    for name in [ 'real']:
        dir_name = os.path.join('/home/yq/domain_net',name)
        path, label = make_dataset_with_labels(dir_name, classes)
        index = np.array(np.arange(len(label)))
        np.random.shuffle(index)
        keep_len = int(len(index)*ratio)
        index = index[:keep_len]
        path = [path[i] for i in index]
        label = [label[i] for i in index]
        print('name', name)
        end = len(label)
        for j in range(keep_len):
            global_data.append(path[j])
            global_label.append(label[j])
        count+=end
    return global_data, global_label
