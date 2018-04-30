import numpy as np
import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from DataStructure import Record
from torch.autograd import Variable
from Util import IndexLinear

class RNNDecoder(nn.Module):
    def __init__(self, u_size, v_size, t_size, opt):
        super(RNNDecoder, self).__init__()
        self.u_size = u_size
        self.v_size = v_size
        self.t_size = t_size
        self.emb_dim_u = opt['emb_dim_u']
        self.emb_dim_v = opt['emb_dim_v']
        self.emb_dim_t = opt['emb_dim_v']
        self.hidden_dim = opt['hidden_dim']
        self.nb_cnt = opt['nb_cnt']
        self.rnn_short = nn.RNN(self.emb_dim_v, self.hidden_dim)
        self.rnn_long = nn.GRU(self.emb_dim_v, self.hidden_dim)
        self.embedder_u = nn.Embedding(self.u_size, self.emb_dim_u)
        self.embedder_v = nn.Embedding(self.v_size, self.emb_dim_v)
        self.embedder_t = nn.Embedding(self.t_size, self.emb_dim_t)
        dim_merged = self.hidden_dim * 2 + self.emb_dim_u + self.emb_dim_t
        self.decoder = IndexLinear(dim_merged, v_size)

    def forward(self, vids, tids, vids_next, tids_next, uids, is_train):
        vids_embeddings = self.embedder_v(vids)
        tids_embeddings = self.embedder_t(tids_next)
        uids_embeddings = self.emb_dim_u(uids)
        hidden_long = self.init_hidden()
        hiddens_long, hidden_long = self.rnn_long(vids_embeddings, hidden_long)


    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_dim))