import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Loss import NSNLLLoss
from Loss import DSSMLoss
import random
import pickle
import torch.optim as optim
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import BallTree

class Work(nn.Module):
    def __init__(self):
        super(Work, self).__init__()
        self.merger_weight_al = []
        for _ in xrange(6):
            self.merger_weight_al.append(nn.Parameter(torch.ones(1, 6) / 6.0))
        feature_weighted = self.merger_weight_al[2] * 0.2
        out = feature_weighted.mean()
        out.backward()
        print self.merger_weight_al[2].grad


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.merger_weight = nn.Parameter(torch.ones(7, 6) / 6.0)
        # print type(self.merger_weight), self.merger_weight.requires_grad

    def forward(self):
        gap_time = 2.6
        gap_time_int = int(gap_time)
        # weight_lower = gap_time_int + 1 - gap_time
        # weight_upper = gap_time - gap_time_int
        # # merger_weight_linear = self.merger_weight[gap_time_int] * weight_lower + self.merger_weight[gap_time_int + 1] * weight_upper
        merger_weight_linear = self.merger_weight[gap_time_int]
        print type(merger_weight_linear), merger_weight_linear.requires_grad
        out = merger_weight_linear.mean()
        print out
        out.backward()
        print merger_weight_linear.grad
        # print self.merger_weight[gap_time_int].grad

# model = Test()
Work()