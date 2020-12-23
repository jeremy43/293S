from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
__all__ = ['Naive']
class Naive(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(Naive, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_drop = nn.Dropout2d()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.bn2 =nn.BatchNorm2d(128)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128*25, 384)
        self.bc3 = nn.BatchNorm1d(384)
        self.fc2 = nn.Linear(384, 192)
        self.bc4 = nn.BatchNorm1d(192)
        self.fc3 = nn.Linear(192, 10)
    def forward(self, x):
        x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
        x = F.relu(self.bn2(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)))
        x = x.view(-1, 128*25)
        #x = F.relu(self.fc1(x))
        x2 = F.relu(self.bc3(self.fc1(x)))
        #x2 = F.dropout(x, training=self.training)
        x1 = self.fc2(x2)
        x = self.bc4(x1)
        #x = F.dropout(x1, training=self.training)
        logit = self.fc3(x)
        #output = x
        output = logit
        #output = F.log_softmax(logit)
        
        #print('output', output.shape)
        #print('fc2 shape', x2.shape)
        f = x.view(x.size(0), -1)
        if not self.training:
            #print('return feature')
            return f,output
        return f,x2,output

