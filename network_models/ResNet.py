from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

__all__ = ['ResNet50', 'ResNet101', 'ResNet50M','ResNet50MA2', 'ResNet18']

class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        y = self.classifier(f)
        if not self.training:
            #print('return feature')
            return f,y
        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        elif self.loss == {'ring'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet101(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet101, self).__init__()
        self.loss = loss
        resnet101 = torchvision.models.resnet101(pretrained=True)
        self.base = nn.Sequential(*list(resnet101.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        elif self.loss == {'ring'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50M(nn.Module):
    """ResNet50 + mid-level features.

    Reference:
    Yu et al. The Devil is in the Middle: Exploiting Mid-level Representations for
    Cross-Domain Instance Matching. arXiv:1711.08106.
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(ResNet50M, self).__init__()
        self.loss = loss
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
        self.classifier = nn.Linear(3072, num_classes)
        #self.mid_classifier = nn.Linear(3072, 288)
        #self.classifier = nn.Linear(288, num_classes)
        self.feat_dim = 3072 # feature dimension

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
        if not self.training:
            #print('evaluate for resnet50m', prelogits.shape)
            return x5c_feat, prelogits
            return combofeat,prelogits

            return final_feature,prelogits

        if self.loss == {'xent'}:
            #print('return feature shape', x5c_feat.shape)
            #print('prelogitss shape', prelogits.shape)
            # The dimension of final_feature is much smaller compared to x5c_feat
            #return final_feature,prelogits
            #print('x5c_feature', x5c_feat.shape)
            return x5c_feat, combofeat, prelogits
            #return x5c_feat,midfeat, prelogits
            return combofeat, prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50MT(nn.Module):
    """ResNet50 + mid-level features + Teachers

    ResNet50M model for teachers
    """
    def __init__(self, num_classes=0, rot_mat=0, loss={'xent'}, **kwargs):
        super(ResNet50MT, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        #self.embedding = nn.Embedding.from_pretrained(rot_mat) rot_mat.shape[1]
        self.embedding = nn.Parameter(rot_mat, requires_grad=False)
        self.classifier = nn.Linear(rot_mat.shape[1], num_classes)
        self.feat_dim = rot_mat.shape[1] # feature dimension

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
        combofeat_rot = torch.mm(combofeat, self.embedding)
        #torch.mm(torch.mm(combofeat,rot_mat),rot_mat.t())
        prelogits = self.classifier(combofeat_rot)

        if not self.training:
            return combofeat_rot,prelogits

        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat_rot
        elif self.loss == {'cent'}:
            return prelogits, combofeat_rot
        elif self.loss == {'ring'}:
            return prelogits, combofeat_rot
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50MTR(nn.Module):
    """ResNet50 + mid-level features + Teachers

    ResNet50M model for teachers
    """
    def __init__(self, num_classes=0, rot_mat=0, loss={'xent'}, **kwargs):
        super(ResNet50MTR, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        #self.embedding = nn.Embedding.from_pretrained(rot_mat) rot_mat.shape[1]
        self.embedding = nn.Parameter(torch.mm(rot_mat, rot_mat.t()), requires_grad=False)
        self.classifier = nn.Linear(3072, num_classes)
        self.feat_dim = 3072 # feature dimension

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
        combofeat_rot = torch.mm(combofeat, self.embedding)
        #torch.mm(torch.mm(combofeat,rot_mat),rot_mat.t())
        prelogits = self.classifier(combofeat_rot)

        if not self.training:
            return combofeat_rot,prelogits

        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat_rot
        elif self.loss == {'cent'}:
            return prelogits, combofeat_rot
        elif self.loss == {'ring'}:
            return prelogits, combofeat_rot
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50MA(nn.Module):
    """ResNet50 + attribute.
    Add one linear layer for attribute classification along with identity classification.
    Multi-task setting
    """
    def __init__(self, num_classesI=0, num_classesA=0, loss={'xent'}, **kwargs):
        super(ResNet50MA, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifierI = nn.Linear(3072, num_classesI)
        self.classifierA = nn.Linear(3072, num_classesA)
        self.feat_dim = 3072 # feature dimension

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



        prelogitsA = self.classifierA(combofeat)
        if not self.training:
            return combofeat,prelogitsA

        prelogitsI = self.classifierI(combofeat)
        
        if self.loss == {'xent'}:
            return prelogitsI
        elif self.loss == {'xent','att'}:
            return prelogitsI, prelogitsA
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50MA1(nn.Module):
    """ResNet50 + attribute.
    Add one linear layer for attribute classification along with identity classification.
    Multi-task setting
    """
    def __init__(self, num_classesI=0, num_classesA=0, loss={'xent'}, **kwargs):
        super(ResNet50MA1, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.fc_i1 = nn.Sequential(nn.Linear(3072,1024))
        self.fc_a1 = nn.Sequential(nn.Linear(1024,512))
        self.fc_a2 = nn.Sequential(nn.Linear(512,256))
        self.classifierI = nn.Linear(1024, num_classesI)
        self.classifierA = nn.Linear(256, num_classesA)
        self.feat_dim = 3072 # feature dimension

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
        feat_i1 = self.fc_i1(combofeat)
        feat_a1 = self.fc_a1(feat_i1)
        feat_a2 = self.fc_a2(feat_a1)
        prelogitsA = self.classifierA(feat_a2)
        if not self.training:
            return feat_i1,prelogitsA
        prelogitsI = self.classifierI(feat_i1)

        
        if self.loss == {'xent'}:
            return prelogitsI
        elif self.loss == {'xent','att'}:
            return prelogitsI, prelogitsA
        elif self.loss == {'xent', 'htri'}:
            return prelogits, feat_i1
        elif self.loss == {'cent'}:
            return prelogits, feat_i1
        elif self.loss == {'ring'}:
            return prelogits, feat_i1
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50MA2(nn.Module):
    """ResNet50 + attribute.
    Add one linear layer for attribute classification along with identity classification.
    Multi-task setting
    """
    def __init__(self, num_classesI=0, num_classesA=0, loss={'xent'}, **kwargs):
        super(ResNet50MA2, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.fc_i1 = nn.Sequential(nn.Linear(3072,512))
        self.fc_a1 = nn.Sequential(nn.Linear(3072,128))
        self.classifierI = nn.Linear(512, num_classesI)
        self.classifierA = nn.Linear(128, num_classesA)
        self.feat_dim = 3072 # feature dimension

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
        feat_i1 = self.fc_i1(combofeat)
        feat_a1 = self.fc_a1(combofeat)
        prelogitsA = self.classifierA(feat_a1)

        if not self.training:
            return feat_i1,prelogitsA
        prelogitsI = self.classifierI(feat_i1)
        if self.loss == {'xent'}:
            #print('ok with resnet', prelogitsA.shape)
            return prelogitsA
            #return prelogitsI
        elif self.loss == {'xent','att'}:
            return prelogitsI, prelogitsA
        elif self.loss == {'xent', 'htri'}:
            return prelogits, feat_i1
        elif self.loss == {'cent'}:
            return prelogits, feat_i1
        elif self.loss == {'ring'}:
            return prelogits, feat_i1
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, loss = {'xent'}, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        f = out.view(out.size(0), -1)
        out = self.linear(f)
        if not self.training:
            return f,out
        if self.loss =={'xent'}:
            return out
        return out


def ResNet18(num_classes=10, loss={'xent'}, **kwargs):
    return ResNet(BasicBlock, [2,2,2,2],loss=loss, num_classes = num_classes)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


# def test():
#     net = ResNet18()
#     y = net(Variable(torch.randn(1,3,32,32)))
#     print(y.size())

# test()

def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    return nn.CrossEntropyLoss()(outputs, labels)


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)
