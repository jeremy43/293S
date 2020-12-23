from __future__ import print_function, absolute_import
import os, csv
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import pickle
import h5py
import scipy.io as sio
#from scipy.misc import imsave
import utils
from PIL import Image




class custom_dataset(object):
    """
    CelebA Attribute Dataset

    """

    def __init__(self, root=None, **kwargs):
        self.dataset_dir = '/home/yq/domain_net'
        self.image_dir = '/home/yq/domain_net'

        label_path = os.path.join(self.dataset_dir,root+'_train.txt')
        with open(label_path, 'r') as txt:
            train_line = txt.readlines()
        test_path = os.path.join(self.dataset_dir, root +'_test.txt')

        with open(test_path,'r') as txt:
            test_line = txt.readlines()
        #self._check_before_run()

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
 
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.image_dir):
            raise RuntimeError("'{}' is not available".format(self.image_dir))

    def _process_dir(self, lines):
        data =[]
        label =[]
        for img_idx, img_info in enumerate(lines):
            img_path = img_info.split(' ', 1)[0]
            cur_label = img_info.split(' ', 1)[1].split()
            img_path = osp.join(self.image_dir, img_path)
            if os.path.exists(img_path):
                pid = np.array(list(map(int, cur_label)))
                img = Image.open(img_path).convert('RGB')
            data.append(img)
            label.append(pid)

        num_imgs = len(data)
        label = np.array(label,dtype=np.int32)
        print('len of label list',len(label))
        return data,label, num_imgs


