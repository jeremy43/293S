#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import sys
sys.path.append('/dataset')
#from dataset.mnist_dataset import Server_Dataset, Custom_Dataset
from dataset.trade_dataset import Server_Dataset, Custom_Dataset
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
#from utils.options import args_parser
#from utils.office_options import args_parser
from utils.mnist_options import args_parser
#from models.noisy_update import LocalUpdate
from autodp import rdp_bank, dp_acct, rdp_acct, privacy_calibrator
from models.Update import LocalUpdate, DA_LocalUpdate
from models.Nets import  ResNet50M,MLP,Alexnet_digit, CNNMnist, CNNCifar, CNNoffice, Naive
from models.Fed import FedAvg, noisy_FedAvg
from models.test import test_img
import models
        



args = args_parser()
delta = 1e-2
noisy_scale =0.06
clip = 0.08
iteration = args.epochs
prob = args.frac
gaussian = lambda x: rdp_bank.RDP_gaussian({'sigma': noisy_scale /clip}, x)
acct = rdp_acct.anaRDPacct()


def privacy_analysis_local():
    """
    compute the local dp
    """
    args = args_parser()
    size =  256# size of local agent
    delta = 1e-4
    noisy_scale =0.5
    iteration = args.epochs * int(size/ args.local_bs)
    prob = args.frac * args.local_bs*1.0/size
    gaussian = lambda x: rdp_bank.RDP_gaussian({'sigma': noisy_scale*clip}, x)
    acct = rdp_acct.anaRDPacct()
    acct.compose_poisson_subsampled_mechanisms(gaussian, prob,coeff = iteration)
    print('local DP', acct.get_eps(delta), delta)

def privacy_analysis():
    """
    compute the local dp
    """
    args = args_parser()
    delta = 1e-3
    noisy_scale =0.5
    clip = 0.7
    iteration = args.epochs 
    prob = args.frac 
    gaussian = lambda x: rdp_bank.RDP_gaussian({'sigma': noisy_scale /clip}, x)
    acct = rdp_acct.anaRDPacct()

    acct.compose_poisson_subsampled_mechanisms(gaussian, prob,coeff = iteration)
    print('global DP', acct.get_eps(delta), delta)


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    print('args dataset', args.dataset)
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        #dataset_train = datasets.MNIST(root='/tmp', train=True, download=True, transform=trans_mnist)
        #dataset_global = Server_Dataset('mnist', transform = trans_mnist, da= True)
        dataset_train = Custom_Dataset('mnist', transform = trans_mnist)
        dataset_test = Server_Dataset('mnist', transform=trans_mnist)
        #dataset_test = datasets.MNIST(root = '/tmp', train=False, download=True, transform=trans_mnist)
        # sample users
        #print('len of target dataset', len(dataset_global))
        #print('use iid distribution data', args.iid)
        """
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
        """
        dict_users = dataset_train.dic_user
    elif args.dataset == 'office':
        trans_office = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = Custom_Dataset('office', transform = trans_office)
        dataset_test = Server_Dataset('office', transform = trans_office)
        dict_users = dataset_train.dic_user
        dataset_global = Server_Dataset('office', transform = trans_office, da= True)
        print('size of testing data', len(dataset_test))
        print('dict_user in office', dict_users.keys())

    elif args.dataset == 'digit':
        #transform_digit = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.485), (0.229))])
        transform_digit = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        dataset_train = Custom_Dataset('digit', transform = transform_digit)
        dataset_global = Custom_Dataset('digit', transform = transform_digit, da= True)
        dataset_test = Server_Dataset('digit', transform = transform_digit)
        dict_users = dataset_train.dic_user
        print('length of training', len(dataset_train))
        print('dic_user[0]', dict_users[0])
        print('size of testing data', len(dataset_test))
        print('dict_user in digit', dict_users.keys())

   
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    #privacy_analysis()
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model =='cnn' and args.dataset =='office':
        #net_glob = models.init_model(name = 'resnet50', num_classes =31, loss = {'xent'}, use_gpu = True)
        net_glob = ResNet50M(args=args).to(args.device)

        #net_glob = CNNoffice(args=args).to(args.device)
    elif args.model =='cnn' and args.dataset == 'digit':
        net_glob = Alexnet_digit(args=args).to(args.device)
        #net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        #net_glob = ResNet50M(args=args).to(args.device)
        net_glob = Alexnet_digit(args=args).to(args.device)
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    acct.compose_poisson_subsampled_mechanisms(gaussian, prob,coeff = 10)
    import math
    nb_batch = int(args.frac * len(dataset_train)/args.local_bs)
    print('len of dataset', len(dataset_train), 'nb_batch', nb_batch)
    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        LR = args.lr
        LR= args.lr / math.pow((1 + 10 * (iter*nb_batch - 1) / (args.epochs*nb_batch)), 0.65)
        if iter % 5 == 0:
            print('cur learning rate', LR)
        for idx in idxs_users:
            if args.da:
                local = DA_LocalUpdate(args=args, dataset=dataset_train,tgt_dataset = dataset_global, idxs=dict_users[idx])
            else:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], iter = LR)
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        #w_glob = FedAvg(w_locals)
        w_glob = noisy_FedAvg(w_locals, w_glob)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        if iter %5 ==0 :
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print("iter {} Testing accuracy: {:.2f}".format(iter, acc_test))
            acct.compose_poisson_subsampled_mechanisms(gaussian, prob,coeff = 5)
            print('current privacy cost', acct.get_eps(delta))

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

