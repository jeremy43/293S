from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb

__all__ = ['discriminator']
class discriminator(nn.Module):
    def __init__(self, feat_dim, num_domain, loss={'xent'}, **kwargs):
        super(discriminator, self).__init__()
        self.feat_dim = feat_dim
        self.num_domain = num_domain
        self.fc1 = nn.Linear(feat_dim, 384)
        self.fc2 = nn.Linear(384, 64)
        self.fc3 = nn.Linear(64, num_domain)
    def forward(self, x):
        x = x.view(-1, self.feat_dim)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        #output = x
        output = F.log_softmax(x)
    
        return output


