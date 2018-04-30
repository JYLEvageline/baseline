import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Loss import NSNLLLoss
import random
import math
import pickle
import torch.optim as optim
from sklearn.neighbors.kde import KernelDensity
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import KDTree
from Util import IndexLinear

class AttentionModel(nn.Module):
    def __init__(self, u_size, v_size, t_size, emb_dim_u=32, emb_dim_v=32, emb_dim_t=16, hidden_dim=32, nb_cnt=15, sampling_list=None, vid_coor_nor=None, vid_pop=None, dropout=0.5, mod=0):
        super(AttentionModel, self).__init__()
        self.u_size = u_size
        self.v_size = v_size
        self.t_size = t_size
        self.emb_dim_u = emb_dim_u
        self.emb_dim_v = emb_dim_v
        self.emb_dim_t = emb_dim_t
        self.hidden_dim = hidden_dim
        self.nb_cnt = nb_cnt
        self.sampling_list = sampling_list
        self.vid_coor_nor = vid_coor_nor
        self.vid_pop = vid_pop
        self.dropout = dropout
        self.mod = mod

        self.tree = KDTree(self.vid_coor_nor.values())
        self.embedder_u = nn.Embedding(self.u_size, self.emb_dim_u)
        self.embedder_v = nn.Embedding(self.v_size, self.emb_dim_v)
        self.embedder_t = nn.Embedding(self.t_size, self.emb_dim_t)
        self.rid_sampling_info = {}
        self.rnn_short = nn.RNNCell(self.emb_dim_v, self.hidden_dim)
        self.rnn_long = nn.GRUCell(self.emb_dim_v, self.hidden_dim)
        self.decoder_h = IndexLinear(self.hidden_dim * 2, v_size)
        self.decoder_t = IndexLinear(self.emb_dim_t, v_size)
        self.decoder_u = IndexLinear(self.emb_dim_u, v_size)
        if self.mod == 0:
            self.merger_weight = nn.Parameter(torch.ones(1, 4) / 4.0)
            # self.merger = nn.Linear(4, 1, bias=False)   # u, t, h, d --> score
        elif self.mod == 1:
            self.merger_weight = nn.Parameter(torch.ones(1, 5) / 5.0)
            # self.merger = nn.Linear(5, 1, bias=False)
        elif self.mod == 2:
            self.merger_weight = nn.Parameter(torch.ones(6, 5) / 5.0)
            # self.merger_al = []
            # for _ in xrange(6):
            #     self.merger_al.append(nn.Linear(5, 1, bias=False))
        # print self.merger_weight
        self.att_dim = self.emb_dim_t + self.hidden_dim * 2
        self.att_M = nn.Parameter(torch.ones(self.att_dim, self.att_dim) / self.att_dim)
        for i in xrange(self.att_dim):
            for j in xrange(self.att_dim):
                if i < self.hidden_dim and j < self.hidden_dim:
                    continue
                if i >= self.hidden_dim and i < self.hidden_dim * 2 and j > self.hidden_dim and j < self.hidden_dim * 2:
                    continue
                if i >= self.hidden_dim * 2 and j > self.hidden_dim * 2:
                    continue
                self.att_M.data[i, j] = 0.0

    def forward(self, records_u, is_train):
        predicted_scores = Variable(torch.zeros(records_u.get_predicting_records_cnt(mod=0), self.nb_cnt + 1)) if is_train else []
        records_al = records_u.get_records(mod=0) if is_train else records_u.get_records(mod=2)
        vids_visited = set([record.vid for record in records_u.get_records(mod=0)])
        emb_u = self.embedder_u(Variable(torch.LongTensor([records_u.uid])).view(1, -1)).view(1, -1)
        emb_u = F.relu(emb_u)
        emb_t_al = Variable(torch.zeros(len(records_al), self.emb_dim_t))
        hidden_long_al = Variable(torch.zeros(len(records_al), self.hidden_dim))
        hidden_short_al = Variable(torch.zeros(len(records_al), self.hidden_dim))
        feature_al = Variable(torch.zeros(len(records_al), self.att_dim))

        hidden_long = self.init_hidden()
        hidden_short = self.init_hidden()
        for idx, record in enumerate(records_u.get_records(mod=0)):  # can only use train data
            if record.is_first:
                hidden_short = self.init_hidden()
            emb_t_al[idx] = F.relu(self.embedder_t(Variable(torch.LongTensor([record.tid])).view(1, -1)).view(1, -1))    # current time embedding
            feature_al[idx] = torch.cat((F.relu(hidden_long), F.relu(hidden_short), emb_t_al[idx].view(1, -1)), 1)
            emb_v = self.embedder_v(Variable(torch.LongTensor([record.vid])).view(1, -1)).view(1, -1)       # feature: current time + previous hiddens
            hidden_long = self.rnn_long(emb_v, hidden_long)
            hidden_short = self.rnn_short(emb_v, hidden_short)
            hidden_long_al[idx] = F.relu(hidden_long)
            hidden_short_al[idx] = F.relu(hidden_short)

        id = 0
        id_vids_true = []
        id_vids = []
        for idx, record in enumerate(records_al):
            if idx >= records_u.test_idx:  # append the states of testing records
                if record.is_first:
                    hidden_short = self.init_hidden()
                emb_t_al[idx] = F.relu(self.embedder_t(Variable(torch.LongTensor([record.tid])).view(1, -1)).view(1, -1))
                feature_al[idx] = torch.cat((F.relu(hidden_long), F.relu(hidden_short), emb_t_al[idx].view(1, -1)), 1)
                emb_v = self.embedder_v(Variable(torch.LongTensor([record.vid])).view(1, -1)).view(1, -1)
                hidden_long = self.rnn_long(emb_v, hidden_long)
                hidden_short = self.rnn_short(emb_v, hidden_short)
                hidden_long_al[idx] = F.relu(hidden_long)
                hidden_short_al[idx] = F.relu(hidden_short)
            if record.is_last or (not is_train and idx < records_u.test_idx):
                continue
            vids_visited.add(record.vid)
            vid_candidates = self.get_vids_candidate(record.rid, vids_visited, record.vid_next, is_train)
            id_vids_true.append(record.vid_next)
            id_vids.append(vid_candidates)
            hidden = torch.cat((hidden_long_al[idx].view(1, -1), hidden_short_al[idx].view(1, -1)), 1)
            scores_u = self.decoder_u(emb_u, Variable(torch.LongTensor(vid_candidates)).view(1, -1))
            scores_t = self.decoder_t(emb_t_al[idx + 1].view(1, -1), Variable(torch.LongTensor(vid_candidates)).view(1, -1))
            scores_h = self.decoder_h(hidden, Variable(torch.LongTensor(vid_candidates)).view(1, -1))
            scores_d_all = self.get_scores_d_all(records_u, idx, vid_candidates, feature_al, is_train)
            if self.mod == 0:
                scores_merge = torch.cat((scores_u, scores_t, scores_h, scores_d_all), 0).t()
                if is_train:
                    # predicted_scores[id] = F.sigmoid(self.merger(scores_merge).t())
                    predicted_scores[id] = F.sigmoid(F.linear(scores_merge, F.relu(self.merger_weight), bias=None).t())
                else:
                    # predicted_scores.append(F.softmax(self.merger(scores_merge).t()))
                    predicted_scores.append(F.softmax(F.linear(scores_merge, F.relu(self.merger_weight), bias=None).t()))
            elif self.mod == 1:
                scores_d_pre = self.get_scores_d_pre(records_u, idx, vid_candidates, feature_al, is_train)
                scores_merge = torch.cat((scores_u, scores_t, scores_h, scores_d_all, scores_d_pre), 0).t()
                if is_train:
                    # predicted_scores[id] = F.sigmoid(self.merger(scores_merge).t())
                    predicted_scores[id] = F.sigmoid(F.linear(scores_merge, F.relu(self.merger_weight), bias=None).t())
                else:
                    # predicted_scores.append(F.softmax(self.merger(scores_merge).t()))
                    predicted_scores.append(F.softmax(F.linear(scores_merge, F.relu(self.merger_weight), bias=None).t()))
            elif self.mod == 2:
                scores_d_pre = self.get_scores_d_pre(records_u, idx, vid_candidates, feature_al, is_train)
                scores_merge = torch.cat((scores_u, scores_t, scores_h, scores_d_all, scores_d_pre), 0).t()
                gap_time = int((records_al[idx + 1].dt - record.dt).total_seconds() / 60 / 30)
                if gap_time >= 6:
                    gap_time = 5
                if is_train:
                    # predicted_scores[id] = F.sigmoid(self.merger_al[gap_time](scores_merge).t())
                    predicted_scores[id] = F.sigmoid(F.linear(scores_merge, F.relu(self.merger_weight[gap_time].view(1, -1)), bias=None).t())
                else:
                    # predicted_scores.append(F.softmax(self.merger_al[gap_time](scores_merge).t()))
                    predicted_scores.append(F.softmax(F.linear(scores_merge, F.relu(self.merger_weight[gap_time].view(1, -1)), bias=None).t()))
            id += 1
        return predicted_scores, id_vids, id_vids_true

    def get_scores_d_pre(self, records_u, idx_cur, vid_candidates, feature_al, is_train):  #id: current record id, want to predict record[id].vid_next
        records_al = records_u.records[0:records_u.test_idx if is_train else idx_cur + 1]
        scores_d_np = np.zeros((1, len(vid_candidates)), dtype=np.float32)
        for idx, vid_candidate in enumerate(vid_candidates):
            scores_d_np[0, idx] = self.get_d_score(records_al[idx_cur].vid, vid_candidate)
        scores_d = Variable(torch.from_numpy(scores_d_np)).view(1, -1)
        return scores_d

    def get_scores_d_all(self, records_u, idx_cur, vid_candidates, feature_al, is_train):  #id: current record id, want to predict record[id].vid_next
        feature_next = feature_al[idx_cur + 1].view(1, -1)
        records_al = records_u.records[0:records_u.test_idx if is_train else idx_cur + 1]
        atten_scores = Variable(torch.zeros(len(records_al)))
        for idx_r, record in enumerate(records_al):
            if idx_r == idx_cur + 1:
                atten_scores.data[idx_r] = float('-inf')
                continue
            feature_r = feature_al[idx_r].view(-1, 1)
            score = torch.mm(torch.mm(feature_next, self.att_M), feature_r)
            atten_scores[idx_r] = score
        atten_scores = F.softmax(atten_scores)

        scores_d = Variable(torch.zeros(1, len(vid_candidates)))
        for idx, vid_candidate in enumerate(vid_candidates):
            score_sum = Variable(torch.zeros([1]))
            for idx_r in xrange(len(records_al)):
                if idx_r == idx_cur + 1:
                    continue
                score = self.get_d_score(records_al[idx_r].vid, vid_candidate)
                score_sum += atten_scores[idx_r] * score
            scores_d[0, idx] = score_sum
        return scores_d

    def get_d_score(self, vid_r, vid_cand):
        coor_diff = self.vid_coor_nor[vid_cand] - self.vid_coor_nor[vid_r]
        return float(np.exp(-np.sqrt(np.sum(coor_diff ** 2))))

    def get_vids_candidate(self, rid, vids_visited, vid_true=None, is_train=True):
        if rid in self.rid_sampling_info:
            vids, probs = self.rid_sampling_info[rid]
        else:
            nbs = set()
            for vid_visited in vids_visited:
                vids = self.tree.query_radius([self.vid_coor_nor[vid_visited]], r=0.07)
                for vid in vids[0]:
                    if (not is_train) or (is_train and vid != vid_true):
                        nbs.add(vid)
            vids = list(nbs)
            probs = np.array([self.vid_pop[vid] for vid in vids], dtype=np.float64)
            probs /= probs.sum()
            self.rid_sampling_info[rid] = (vids, probs)
        if is_train:
            id_cnt = np.random.multinomial(self.nb_cnt, probs)
            vid_candidates = [vid_true]
            for id, cnt in enumerate(id_cnt):
                for _ in range(cnt):
                    vid_candidates.append(vids[id])
            return vid_candidates
        else:
            return vids

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_dim))

def train(root_path, dataset, n_iter=500, iter_start=0, mod=0):
    torch.manual_seed(0)
    random.seed(0)
    dl = pickle.load(open(root_path + 'dl_' + dataset + '.pk', 'rb'))
    for _, records_u in dl.uid_records.items():
        records_u.summarize()
    model = AttentionModel(dl.nu, dl.nv, dl.nt, sampling_list=dl.sampling_list, vid_coor_nor=dl.vid_coor_nor, vid_pop=dl.vid_pop, mod=mod)
    if iter_start != 0:
        model.load_state_dict(torch.load(root_path + 'model_attention_nobias_' + str(mod) + '_' + str(iter_start) + '.md'))
    optimizer = optim.Adam(model.parameters())
    criterion = NSNLLLoss()
    uids = dl.uid_records.keys()
    for iter in range(iter_start + 1, n_iter + 1):
        print_loss_total = 0
        random.shuffle(uids)
        for idx, uid in enumerate(uids):
            records_u = dl.uid_records[uid]
            optimizer.zero_grad()
            predicted_probs, _, _ = model(records_u, is_train=True)
            loss = criterion(predicted_probs)
            loss.backward()
            print_loss_total += loss.data[0]
            optimizer.step()
            if idx % 50 == 0:
                print 'uid: \t%d\tloss: %f' % (idx, print_loss_total)
        print iter, print_loss_total
        if iter % 1 == 0:
            torch.save(model.state_dict(), root_path + 'model_attention_nobias_' + str(mod) + '_' + str(iter) + '.md')

def test(root_path, dataset, iter_start=0, mod=0):
    torch.manual_seed(0)
    random.seed(0)
    dl = pickle.load(open(root_path + 'dl_' + dataset + '.pk', 'rb'))
    for _, records_u in dl.uid_records.items():
        records_u.summarize()
    for iter in range(iter_start, 0, -5):
        model = AttentionModel(dl.nu, dl.nv, dl.nt, sampling_list=dl.sampling_list, vid_coor_nor=dl.vid_coor_nor,
                               vid_pop=dl.vid_pop, mod=mod)
        if iter_start != 0:
            model.load_state_dict(torch.load(root_path + 'model_attention_nobias_' + str(mod) + '_' + str(iter) + '.md'))
        hits = np.zeros(3)
        cnt = 0
        for uid, records_u in dl.uid_records.items():
            id_scores, id_vids, vids_true = model(records_u, is_train=False)
            for idx in range(len(id_vids)):
                probs_sorted, vid_sorted = torch.sort(id_scores[idx].view(-1), 0, descending=True)
                vid_ranked = [id_vids[idx][id] for id in vid_sorted.data]
                cnt += 1
                for j in range(min(len(vid_ranked), 10)):
                    if vids_true[idx] == vid_ranked[j]:
                        if j == 0:
                            hits[0] += 1
                        if j < 5:
                            hits[1] += 1
                        if j < 10:
                            hits[2] += 1
            if (uid + 1) % 1 == 0:
                print (uid + 1), hits / cnt
        hits /= cnt
        print 'iter:', iter, 'hits: ', hits, 'cnt: ', cnt

