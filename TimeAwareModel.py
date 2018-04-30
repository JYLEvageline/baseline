import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Util import IndexLinear
from Loss import NSNLLLoss
import string
import random
import pickle
import math
import torch.optim as optim
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import BallTree

class TimeAwareModel(nn.Module):
    def __init__(self, u_size, v_size, t_size, emb_dim=50, nb_cnt=100, sampling_list=None, vid_coor_rad=None, dropout=0.5, mod=0, mod_merge=0):
        super(TimeAwareModel, self).__init__()
        self.emb_dim = emb_dim
        self.u_size = u_size
        self.v_size = v_size
        self.t_size = t_size
        self.nb_cnt = nb_cnt
        self.dropout = dropout
        self.sampling_list = sampling_list
        self.vid_coor_rad = vid_coor_rad
        if self.vid_coor_rad is not None:
            self.tree = BallTree(vid_coor_rad.values(), leaf_size=40, metric='haversine')
            self.dist_metric = DistanceMetric.get_metric('haversine')
        self.uid_rid_nbs = {}
        for uid in range(0, u_size):
            self.uid_rid_nbs[uid] = {}
        self.mod = mod
        self.mod_merge = mod_merge
        self.rnn_short = nn.RNNCell(emb_dim, emb_dim)
        self.rnn_long = nn.GRUCell(emb_dim, emb_dim)
        self.embedder_u = nn.Embedding(u_size, emb_dim)
        self.embedder_v = nn.Embedding(v_size, emb_dim)
        if mod == 0:    #mod 0: cat(u, long, short)
            self.decoder = IndexLinear(emb_dim * 3, v_size)
        elif mod == 1:    #mod 1: cat(u, merge)
            self.rnn_merge = MergeRNNCell(emb_dim, emb_dim, mod_merge=mod_merge)
            self.decoder = IndexLinear(emb_dim * 2, v_size)
        elif mod == 2:  #mod 2: cat(u, t_next, long, short)
            self.embedder_t = nn.Embedding(t_size, emb_dim)
            self.decoder = IndexLinear(emb_dim * 4, v_size)
        elif mod == 3:    #mod 1: cat(u, t_next, merge)
            self.rnn_merge = MergeRNNCell(emb_dim, emb_dim, mod_merge=mod_merge)
            self.decoder = IndexLinear(emb_dim * 3, v_size)
        elif mod == 4:
            self.embedder_t = nn.Embedding(t_size, emb_dim)
            self.decoder = IndexLinear(emb_dim * 4, v_size)
            self.embedder_gap_time = nn.Embedding(12, 2)
            self.merger = nn.Linear(2, 1)
        elif mod == 5:
            self.embedder_t = nn.Embedding(t_size, emb_dim)
            self.decoder = IndexLinear(emb_dim * 4, v_size)
            self.embedder_gap_time = nn.Embedding(12, 2)
            self.merger = nn.Linear(2, 1)

    def forward(self, records_u, is_train):
        records_al = records_u.get_records(mod=0) if is_train else records_u.get_records(mod=2)
        emb_u = F.dropout(self.embedder_u(Variable(torch.LongTensor([records_u.uid])).view(1, -1)).view(1, -1)) if is_train else self.embedder_u(Variable(torch.LongTensor([records_u.uid])).view(1, -1)).view(1, -1)
        hidden_long = self.init_hidden()
        if self.mod in {1, 3}:
            hidden_merge = self.init_hidden()
        predicted_scores = []
        # predicted_scores = Variable(
        #     torch.zeros(records_u.get_predicting_records_cnt(mod=0), self.nb_cnt + 1)) if is_train else Variable(
        #     torch.zeros(records_u.get_predicting_records_cnt(mod=2), self.v_size))
        vids_true = []
        idx = 0
        session_start_rid = 0
        vids_visited = set()
        if self.mod in {4, 5}:
            id_vids = []
        for rid, record in enumerate(records_al[0: len(records_al) - 1]):
            if record.is_first:
                hidden_short = self.init_hidden()
                session_start_rid = rid
            vids_visited.add(record.vid)
            emb_v = F.dropout(self.embedder_v(Variable(torch.LongTensor([record.vid])).view(1, -1)).view(1, -1)) if is_train else self.embedder_v(Variable(torch.LongTensor([record.vid])).view(1, -1)).view(1, -1)
            hidden_long = self.rnn_long(emb_v, hidden_long)
            hidden_short = self.rnn_short(emb_v, hidden_short)
            if record.is_last:
                continue
            if self.mod == 0:
                hidden = torch.cat((emb_u.view(-1, self.emb_dim), hidden_long.view(-1, self.emb_dim), hidden_short.view(-1, self.emb_dim)), 1)
            elif self.mod == 1:
                if self.mod_merge == 2: #consider gap time
                    gap_time = (records_al[rid + 1].dt - record.dt).total_seconds() / 3600.0
                    hidden_merge = self.rnn_merge(hidden_long, hidden_short, hidden_merge, rid - session_start_rid, gap_time=gap_time)
                else:
                    hidden_merge = self.rnn_merge(hidden_long, hidden_short, hidden_merge, rid - session_start_rid)
                hidden = torch.cat(
                    (emb_u.view(-1, self.emb_dim), hidden_merge.view(-1, self.emb_dim)), 1)
            elif self.mod in {2, 4, 5}:
                emb_t = F.dropout(self.embedder_v(Variable(torch.LongTensor([records_al[rid+1].tid])).view(1, -1)).view(1, -1)) if is_train else self.embedder_v(Variable(torch.LongTensor([records_al[rid+1].tid])).view(1, -1)).view(1, -1)
                hidden = torch.cat((emb_u.view(-1, self.emb_dim), emb_t.view(-1, self.emb_dim), hidden_long.view(-1, self.emb_dim),
                                    hidden_short.view(-1, self.emb_dim)), 1)
            elif self.mod == 3:
                emb_t = F.dropout(self.embedder_v(Variable(torch.LongTensor([records_al[rid+1].tid])).view(1, -1)).view(1, -1)) if is_train else self.embedder_v(Variable(torch.LongTensor([records_al[rid+1].tid])).view(1, -1)).view(1, -1)
                if self.mod_merge == 2: #consider gap time
                    gap_time = (records_al[rid + 1].dt - record.dt).total_seconds() / 3600.0
                    hidden_merge = self.rnn_merge(hidden_long, hidden_short, hidden_merge, rid - session_start_rid,
                                                  gap_time=gap_time)
                else:
                    hidden_merge = self.rnn_merge(hidden_long, hidden_short, hidden_merge, rid - session_start_rid)
                hidden = torch.cat((emb_u.view(-1, self.emb_dim), emb_t.view(-1, self.emb_dim), hidden_merge.view(-1, self.emb_dim)), 1)
            if self.mod not in {4, 5}:
                if is_train:
                    vids_true.append(record.vid_next)
                    vid_candidates = self.get_vid_candidates(record.vid_next)
                    hidden = F.dropout(hidden)
                    output = self.decoder(hidden, vid_candidates.view(1, -1))
                    predicted_scores.append(F.softmax(output))
                    idx += 1
                else:
                    if rid >= records_u.test_idx:
                        vids_true.append(record.vid_next)
                        output = self.decoder(hidden)
                        predicted_scores.append(F.softmax(output))
                        idx += 1
            else:   #mod = 4 or 5
                if is_train:
                    vids_true.append(record.vid_next)
                    vid_candidates = self.get_vid_candidates_dis(records_u.uid, rid, record.vid_next, vids_visited, True)
                    gap_time = int((records_al[rid + 1].dt - record.dt).total_seconds() / 60 / 30)
                    emb_gap_time = self.embedder_gap_time(Variable(torch.LongTensor([gap_time])).view(1, -1)).view(1, -1)
                    dis_weight = F.softmax(emb_gap_time)
                    id_scores_tmp = self.decoder(hidden, vid_candidates.view(1, -1)).view(-1)
                    id_score_dist = Variable(torch.zeros(self.nb_cnt + 1))
                    for vid_idx, vid_candidate_var in enumerate(vid_candidates):
                        vid_candidate =  vid_candidate_var.data[0]
                        dis_pre = self.dist_metric.pairwise([self.vid_coor_rad[vid_candidate], self.vid_coor_rad[record.vid]])[0][1]
                        if self.mod == 4:
                            dis_min = dis_pre
                            for vid_visited in vids_visited:
                                dis_min = min((dis_min, self.dist_metric.pairwise(
                                    [self.vid_coor_rad[vid_candidate], self.vid_coor_rad[vid_visited]])[0][1]))
                            dis_vec = Variable(torch.FloatTensor([dis_pre, dis_min])).view(1, -1)
                            id_score_dist[vid_idx] = torch.mm(dis_vec, dis_weight.view(-1, 1))
                        else:
                            id_score_dist[vid_idx] = Variable(torch.FloatTensor([dis_pre])).view(1, -1)
                    id_scores = torch.cat((id_scores_tmp.unsqueeze(1), id_score_dist.unsqueeze(1)), 1)
                    predicted_scores.append(F.softmax(self.merger(id_scores).view(-1)).view(1, -1))
                    id_vids.append(vid_candidates)
                    idx += 1
                else:
                    if rid >= records_u.test_idx:
                        vids_true.append(record.vid_next)
                        vid_candidates = self.get_vid_candidates_dis(records_u.uid, rid, record.vid_next, vids_visited,
                                                                     True, record.vid)
                        gap_time = int((records_al[rid + 1].dt - record.dt).total_seconds() / 60 / 30)
                        emb_gap_time = self.embedder_gap_time(Variable(torch.LongTensor([gap_time])).view(1, -1)).view(
                            1, -1)
                        dis_weight = F.softmax(emb_gap_time)
                        id_scores_tmp = self.decoder(hidden, vid_candidates.view(1, -1)).view(-1)

                        id_score_dist = Variable(torch.zeros(vid_candidates.size(0)))
                        for vid_idx, vid_candidate_var in enumerate(vid_candidates):
                            vid_candidate = vid_candidate_var.data[0]
                            dis_pre = self.dist_metric.pairwise(
                                [self.vid_coor_rad[vid_candidate], self.vid_coor_rad[record.vid]])[0][1]
                            dis_min = dis_pre
                            for vid_visited in vids_visited:
                                dis_min = min((dis_min, self.dist_metric.pairwise(
                                    [self.vid_coor_rad[vid_candidate], self.vid_coor_rad[vid_visited]])[0][1]))
                            dis_vec = Variable(torch.FloatTensor([dis_pre, dis_min])).view(1, -1)
                            id_score_dist[vid_idx] = torch.mm(dis_vec, dis_weight.view(-1, 1))
                        id_scores = torch.cat((id_scores_tmp.unsqueeze(1), id_score_dist.unsqueeze(1)), 1)
                        predicted_scores.append(F.softmax(self.merger(id_scores).view(-1)).view(1, -1))
                        id_vids.append(vid_candidates)
                        idx += 1
        if self.mod in {4, 5}:
            return predicted_scores, id_vids, vids_true
        return predicted_scores, None, vids_true

    def get_vid_candidates(self, vid):
        reject = set()
        reject.add(vid)
        vid_candidates = [vid]
        while len(reject) <= self.nb_cnt:
            vid_candidate = self.sampling_list[random.randint(0, len(self.sampling_list) - 1)]
            if vid_candidate not in reject:
                reject.add(vid_candidate)
                vid_candidates.append(vid_candidate)
        return Variable(torch.LongTensor(vid_candidates))

    def get_vid_candidates_dis(self, uid, rid, vid_true, vids_visited, is_train, vid_current=None):
        if rid in self.uid_rid_nbs[uid]:
            vids_visited_nb = self.uid_rid_nbs[uid][rid]
        else:
            vids_visited_nb = set()
            for vid in vids_visited:
                dist, ids = self.tree.query([self.vid_coor_rad[vid]], 300)
                for i in range(0, len(ids[0])):
                    vid_candidate = ids[0, i]
                    vids_visited_nb.add(vid_candidate)
            self.uid_rid_nbs[uid][rid] = vids_visited_nb
        if is_train:
            samp_list = list(vids_visited_nb)
            reject = set()
            reject.add(vid_true)
            vid_candidates = [vid_true]
            while len(reject) <= self.nb_cnt:
                vid_candidate = samp_list[random.randint(0, len(samp_list) - 1)]
                if vid_candidate not in reject:
                    reject.add(vid_candidate)
                    vid_candidates.append(vid_candidate)
            return Variable(torch.LongTensor(vid_candidates))#, vids_dist
        else:
            vids_visited_nb.clear()
            dist, ids = self.tree.query([self.vid_coor_rad[vid_current]], 500)
            for i in range(0, len(ids[0])):
                vid_candidate = ids[0, i]
                vids_visited_nb.add(vid_candidate)
            return Variable(torch.LongTensor(list(vids_visited_nb)))#, vids_dist

    def init_hidden(self):
        return Variable(torch.zeros(1, self.emb_dim))


class MergeRNNCell(nn.Module):
    def __init__(self, input_dim, emb_dim, mod_merge=0):
        super(MergeRNNCell, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.mod_merge = mod_merge

        self.weight_ii_l = nn.Parameter(torch.FloatTensor(emb_dim, input_dim))
        self.weight_ir_l = nn.Parameter(torch.FloatTensor(emb_dim, input_dim))
        self.weight_in_l = nn.Parameter(torch.FloatTensor(emb_dim, input_dim))

        self.weight_ii_s = nn.Parameter(torch.FloatTensor(emb_dim, input_dim))
        self.weight_ir_s = nn.Parameter(torch.FloatTensor(emb_dim, input_dim))
        self.weight_in_s = nn.Parameter(torch.FloatTensor(emb_dim, input_dim))

        self.weight_hi = nn.Parameter(torch.FloatTensor(emb_dim, emb_dim))
        self.weight_hr = nn.Parameter(torch.FloatTensor(emb_dim, emb_dim))
        self.weight_hn = nn.Parameter(torch.FloatTensor(emb_dim, emb_dim))

        self.bias_ii = nn.Parameter(torch.FloatTensor(emb_dim))
        self.bias_ir = nn.Parameter(torch.FloatTensor(emb_dim))
        self.bias_in = nn.Parameter(torch.FloatTensor(emb_dim))

        self.bias_hi = nn.Parameter(torch.FloatTensor(emb_dim))
        self.bias_hr = nn.Parameter(torch.FloatTensor(emb_dim))
        self.bias_hn = nn.Parameter(torch.FloatTensor(emb_dim))
        self.reset_parameters()
        if self.mod_merge == 1:
            self.linear_length = nn.Linear(1, 1)
            self.linear_length.weight.data.fill_(0.05)
            self.linear_length.bias.data.fill_(0.5)
        elif self.mod_merge == 2:
            self.linear_length = nn.Linear(2, 1)
            self.linear_length.weight.data[0,0]= 0.05
            self.linear_length.weight.data[0, 1] = -0.05
            self.linear_length.bias.data[0] = 0.5

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, hidden_long, hidden_short, hidden_merge, session_len, gap_time=-1):
        if self.mod_merge == 0:
            weight = Variable(torch.FloatTensor([np.tanh(0.333 * session_len)])).view(1, -1)
        elif self.mod_merge == 1:
            weight = self.linear_length(Variable(torch.FloatTensor([session_len])).view(1, -1)).clamp(min=0, max=1)
        elif self.mod_merge == 2:
            weight = self.linear_length(Variable(torch.FloatTensor([session_len, gap_time])).view(1, -1)).clamp(min=0, max=1)
        # print 'len:', session_len, 'gap: ', gap_time, 'weight: ', weight, self.linear_length.weight, self.linear_length.bias
        # raw_input()
        i = F.sigmoid(F.linear(hidden_long, self.weight_ii_l) * (1 - weight).expand_as(hidden_long) +
                      F.linear(hidden_short, self.weight_ii_s) * weight.expand_as(hidden_short)  + self.bias_ii +
                      F.linear(hidden_merge, self.weight_hi, self.bias_hi))
        r = F.sigmoid(F.linear(hidden_long, self.weight_ir_l) * (1 - weight).expand_as(hidden_long) +
                      F.linear(hidden_short, self.weight_ir_s) * weight.expand_as(hidden_short) + self.bias_ir +
                      F.linear(hidden_merge, self.weight_hr, self.bias_hr))
        n = F.tanh(F.linear(hidden_long, self.weight_in_l) * (1 - weight).expand_as(hidden_long)  +
                   F.linear(hidden_short, self.weight_in_s) * weight.expand_as(hidden_short) + self.bias_in +
                   r * F.linear(hidden_merge, self.weight_hn, self.bias_hn))
        hidden_merge_new = (1 - i) * n + i * hidden_merge
        return hidden_merge_new

def train(root_path, emb_dim=50, nb_cnt=100, n_iter=500, iter_start=0, dropout=0.5, mod=0, mod_merge=0, dataset=None):
    torch.manual_seed(0)
    random.seed(0)
    dl = pickle.load(open(root_path + 'dl_' +dataset+'.pk', 'rb'))
    model = TimeAwareModel(dl.nu, dl.nv, emb_dim, nb_cnt=nb_cnt, sampling_list=dl.sampling_list, vid_coor_rad=dl.vid_coor_rad, dropout=dropout, mod=mod, mod_merge=mod_merge)
    if iter_start != 0:
        model.load_state_dict(torch.load(root_path + 'model_tam_' + str(mod) + '_' + str(mod_merge) + '_' + str(iter_start) + '.md'))
    optimizer = optim.Adam(model.parameters())
    criterion = NSNLLLoss()
    uids = dl.uid_records.keys()
    for iter in range(iter_start + 1, n_iter + 1):
        print_loss_total = 0
        # random.shuffle(uids)
        for idx, uid in enumerate(uids):
            records_u = dl.uid_records[uid]
            optimizer.zero_grad()
            predicted_probs, _, _ = model(records_u, True)
            loss = criterion(predicted_probs)
            loss.backward()
            print_loss_total += loss.data[0]
            optimizer.step()
            if idx % 100 == 0:
                print 'uid: \t%d\tloss: %f' % (idx, print_loss_total)
        print iter, print_loss_total
        if iter % 5 == 0:
            torch.save(model.state_dict(), root_path + 'model_tam_' + str(mod) + '_' + str(mod_merge) + '_' + str(iter) + '.md')


def test(root_path, emb_dim=50, nb_cnt=100, iter_start=0, mod=0, mod_merge=0, dataset=None):
    dl = pickle.load(open(root_path + 'dl_' +dataset+'.pk', 'rb'))
    for iter in range(iter_start, 0, -5):
        model = TimeAwareModel(dl.nu, dl.nv, emb_dim, nb_cnt=nb_cnt, sampling_list=dl.sampling_list, vid_coor_rad=dl.vid_coor_rad, mod=mod,
                               mod_merge=mod_merge)
        if iter_start != 0:
            model.load_state_dict(
                torch.load(root_path + 'model_tam_' + str(mod) + '_' + str(mod_merge) + '_' + str(iter) + '.md'))
        hits = np.zeros(3)
        cnt = 0
        for uid, records_u in dl.uid_records.items():
            id_scores, id_vids, vids_true = model(records_u, False)
            for idx in range(0, len(id_vids)):
                probs_sorted, vid_sorted = torch.sort(id_scores[idx].view(-1), 0, descending=True)
                # print 'probs_sorted: ', probs_sorted
                # print 'vid_sorted: ', vid_sorted
                # print 'id_vids: ', id_vids[idx]
                # print 'vids_true: ', vids_true
                vid_ranked = [id_vids[idx].data[id] for id in vid_sorted.data]
                cnt += 1
                for j in range(10):
                    if vids_true[idx] == vid_ranked[j]:
                        if j == 0:
                            hits[0] += 1
                        if j < 5:
                            hits[1] += 1
                        if j < 10:
                            hits[2] += 1
            if (uid + 1) % 10 == 0:
                print (uid + 1), hits / cnt
        hits /= cnt
        print 'iter:', iter, 'hits: ', hits, 'cnt: ', cnt