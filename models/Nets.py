#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 84)
        self.fc2 = nn.Linear(84, args.num_classes)
        self.da = args.da
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        #print('x shape' ,x.shape)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        output = self.fc2(x)
        if self.da:
            return x, output
        return output


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 4*4 for mnist or 5*5 for svhn (32)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)
        self.da = args.da
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print('x shape', x.shape) # 5*5 used for 32 and 4*4 used for 28
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        feature = F.relu(self.fc2(x))
        x = self.fc3(feature)
        if self.da:
            return feature, x
        else:
            return x

class Naive(nn.Module):
    def __init__(self, args):
        super(Naive, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv1_drop = nn.Dropout2d()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        #self.fc1 = nn.Linear(128*53*53, 384)
        self.fc1 = nn.Linear(128*25, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)
        self.da = args.da
    def forward(self, x):
        x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
        x = F.relu(self.bn2(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)))
        x = x.view(-1, 128*25)
        x2 = F.relu(self.bc3(self.fc1(x))) 
        x1 = self.fc2(x2)
        x = self.bc4(x1) 
        logit = self.fc3(x)
        output = logit
        f = x.view(x.size(0), -1)
        if self.da:
            return f, output
        else:
            return output


class CNNoffice(nn.Module):
    def __init__(self, args):
        super(CNNoffice, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*61*61, 120)
        #self.fc1 = nn.Linear(16 * 72 * 72, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes) 
        self.da = args.da
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print('x.shape',x.shape)
        x = x.view(-1, 16*61*61)
        #x = x.view(-1, 16 * 72 * 72)
        feature = F.relu(self.fc1(x))
        x = F.relu(self.fc2(feature))
        x = self.fc3(feature)
        if self.da:
            return feature, x
        return x

class CNNcaltech(nn.Module):
    def __init__(self, args):
        super(CNNcaltech, self).__init__()
        self.da = args.da
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv1_drop = nn.Dropout2d()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128*61*61, 384)
        #self.fc1 = nn.Linear(128*25, 384)
        self.fc2 = nn.Linear(384, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #print('x shape', x.shape)
        x = x.view(-1, 128*61*61)
        x2 = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x2))
        logit = self.fc3(x)
        if self.da:
            return x, logit
        return x






class Discriminator(nn.Module):
    def __init__(self, feat_dim, num_domain, loss={'xent'}, **kwargs):
        super(Discriminator, self).__init__()
        self.feat_dim = feat_dim
        self.num_domain = num_domain
        self.fc1 = nn.Linear(feat_dim, 64)
        self.fc2 = nn.Linear(384, 64)
        self.fc3 = nn.Linear(64, num_domain)
    def forward(self, x):
        x = x.view(-1, self.feat_dim)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        #x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        #output = x
        output = F.log_softmax(x)

        return output

class Alexnet(nn.Module):

    def __init__(self, args):
        super(Alexnet, self).__init__()
        model = torchvision.models.alexnet(pretrained = True)
        self.feature = nn.Sequential(*list(model.features.children()))
        self.fc = nn.Linear(256*7*7, 384)
        self.args = args
        self.fc2 = nn.Linear(384, args.num_classes)
        #self.classifier = nn.Sequential(
        #    *[list(model.classifier.children())[i] for i in [1, 2, 4, 5]], nn.Linear(4096,10))        
        self.da = args.da
    def forward(self, x):
        feature = self.feature(x)
        #2 *2 used for face or digit with 64
        feature = feature.view(-1, 256*7*7)
        feature_1 =self.fc(feature)
        feature_1 = F.relu(feature_1)
        x  = self.fc2(feature_1)
        if self.da:
            return feature_1, x
        return x



class Alexnet_digit(nn.Module):

    def __init__(self, args):
        super(Alexnet_digit, self).__init__()
        model = torchvision.models.alexnet(pretrained = True)
        self.feature = nn.Sequential(*list(model.features.children()))
        self.fc = nn.Linear(256*1, 384) #domain net use 7*11
        #self.classifier = nn.Sequential(
        #    *[list(model.classifier.children())[i] for i in [1, 2, 4, 5]], nn.Linear(4096,10))        
        self.fc2 = nn.Linear(384, args.num_classes)
        self.da = args.da
    def forward(self, x):
        #print('x shape', x.shape)
        feature = self.feature(x)
        #print('feature shape', feature.shape)
        feature = feature.view(-1, 256*1*1)
        #feature = feature.view(-1, 256*7*11)
        feature_1 =self.fc(feature)
        feature_1 = F.relu(feature_1)
        x  = self.fc2(feature_1)
        if self.da:
            return feature_1, x
        return x




class ResNet50M(nn.Module):
    """ResNet50 + mid-level features.

    Reference:
    Yu et al. The Devil is in the Middle: Exploiting Mid-level Representations for
    Cross-Domain Instance Matching. arXiv:1711.08106.
    """
    def __init__(self, args):
        super(ResNet50M, self).__init__()
        #resnet50 = torchvision.models.resnet50(pretrained=True)
        resnet50 = torchvision.models.resnet50(pretrained=True)
        #print('The nesnet now is not pretraiend')
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(3072, args.num_classes)
        #self.mid_classifier = nn.Linear(3072, 288)
        #self.classifier = nn.Linear(288, num_classes)
        self.feat_dim = 3072 # feature dimension
        self.args = args
        self.da= args.da
    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        combofeat = torch.cat((x5c_feat, midfeat), dim=1)
        # print('combofeat shape', combofeat.shape)
        #final_feature = self.mid_classifier(combofeat)
        #prelogits = self.classifier(final_feature)
        prelogits = self.classifier(combofeat)
        #prelogits = F.log_softmax(prelogits, dim=1)
        if self.da:
            return combofeat, prelogits
        return prelogits


