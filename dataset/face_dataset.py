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
size = 256
import numpy as np
dataset_path = 'face_local_data.pkl'
test_path = 'face_test_data.pkl'

class custom_dataset(object):
    #build a customized dataset

    def __init__(self, root=None,transform=None, **kwargs):
        self.dataset_dir = '/home/yq/clean_kNN/list'
        self.image_dir = '/home/yq/new_celeba'
        if transform is not None:
            self.transform = transform
        label_path = os.path.join(self.dataset_dir,'celeba_2d_train_filelist.txt')
        with open(label_path, 'r') as txt:
            train_line = txt.readlines()
        test_path = os.path.join(self.dataset_dir, 'celeba_2d_test_filelist.txt')

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
                cur_label =  np.array(list(map(int, cur_label)))
                label.append(cur_label)

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


def load_dataset():
    #assert(len(classes) == 10)
    idx = 0
    count = 0
    local_img = []
    local_label = []
    dic_users = {}
    cur_dataset = custom_dataset()
    path = cur_dataset.train_data
    label = cur_dataset.train_label
    nb_teachers = 300
    batch_len = int(len(path)/nb_teachers)
    for idx in range(300):
        start = idx * batch_len
        end = min((idx + 1) * batch_len, len(path)-1)
        index = np.array(np.arange(start, end))
        for j in range(start, end):
            local_img.append(path[j])
            local_label.append(label[j])
        dic_users[idx] = index


    print('total number of agents', idx+1)
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
            dic_user, img_set, label_set = load_dataset()
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
    global_data = []
    global_label = []
    cur_dataset = custom_dataset()
    global_data = cur_dataset.test_data
    global_label = cur_dataset.test_label
    return global_data, global_label
