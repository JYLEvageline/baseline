import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class IndexLinear(nn.Linear):
    def forward(self, embedding, indices=None):
        if indices is None:
            return super(IndexLinear, self).forward(embedding)
        embedding = embedding.unsqueeze(1)
        weight = torch.index_select(self.weight, 0, indices.view(-1)).view(indices.size(0), indices.size(1), -1).transpose(1, 2)
        bias = torch.index_select(self.bias, 0, indices.view(-1)).view(indices.size(0), 1, indices.size(1))
        out = torch.baddbmm(1, bias, 1, embedding, weight)  # bias + embedding * weight
        return out.squeeze().unsqueeze(0)

    def reset_parameters(self):
        init_range = 0.1
        self.bias.data.fill_(0)
        self.weight.data.uniform_(-init_range, init_range)

