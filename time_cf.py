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

class TimeAwareCF(nn.Module):
    def __init__(self, v_size, t_size, emb_dim_t=16, distance_threshold=0.07, vid_coor_nor=None, vid_pop=None, nb_cnt=15, adapt_bandwidth=False, bandwidth_global=0.07, adapt_nn=5):
        super(TimeAwareCF, self).__init__()
        self.v_size = v_size
        self.t_size = t_size
        self.emb_dim_t = emb_dim_t
        self.vid_coor_nor = vid_coor_nor
        self.vid_pop = vid_pop
        self.tree = KDTree(self.vid_coor_nor.values())
        self.embedder_t = nn.Embedding(self.t_size, self.emb_dim_t)
        self.bandwidth = bandwidth_global
        self.sampling_info = {}
        self.dist_threshold = distance_threshold
        self.nb_cnt=nb_cnt
        self.rid_sampling_info = {}
        self.vid_band = {}
        self.adapt_bandwidth(adapt_bandwidth, adapt_nn)
        # print self.vid_band
        # raw_input()

    def adapt_bandwidth(self, adapt_bandwidth, adapt_nn):
        if not adapt_bandwidth:
            for vid in xrange(self.v_size):
                self.vid_band[vid] = self.bandwidth
        else:
            bandwidth_total = 0.0
            for vid in xrange(self.v_size):
                dists, ids = self.tree.query([self.vid_coor_nor[vid]], adapt_nn + 1)
                self.vid_band[vid] = dists[0, adapt_nn]
                # print dists[0, k]
                bandwidth_total += dists[0, adapt_nn]
            bandwidth_total /= self.v_size
            # print bandwidth_total
            # raw_input()

    def direct_test(self, records_u):
        id_vids_true = []
        id_vids = []
        predicted_scores = []
        vids_visited = set([record.vid for record in records_u.get_records(mod=0)])
        records_al = records_u.get_records(mod=1)
        idx = 0
        for id, record in enumerate(records_al):
            vids_visited.add(record.vid)
            if record.is_last:
                continue
            id_vids_true.append(record.vid_next)
            vid_candidates = self.get_vids_candidate(record.rid, None, vids_visited, False)
            score_raw = Variable(torch.zeros(1, len(vid_candidates)))
            for i, vid_candidate in enumerate(vid_candidates):
                score_raw[0, i] = self.get_kde_score_decay(records_al, id, vid_candidate)
            predicted_scores.append(F.softmax(score_raw))
            id_vids.append(vid_candidates)
            idx += 1
        return predicted_scores, id_vids, id_vids_true

    def forward(self, records_u, is_train):
        id_vids_true = []
        id_vids = []
        predicted_scores = Variable(
            torch.zeros(records_u.get_predicting_records_cnt(mod=0), self.nb_cnt + 1)) if is_train else []
        records_al = records_u.get_records(mod=0) if is_train else records_u.get_records(mod=1)
        vids_visited = set([record.vid for record in records_u.get_records(mod=0)])
        emb_t_al = Variable(torch.zeros(len(records_al), self.emb_dim_t))
        for id, record in enumerate(records_al):
            emb_t_al[id] = self.embedder_t(Variable(torch.LongTensor([record.tid])).view(1, -1)).view(1, -1)
        idx = 0
        for id, record in enumerate(records_al):
            vids_visited.add(record.vid)
            if record.is_last:
                continue
            id_vids_true.append(record.vid_next)
            if is_train:
                vid_candidates = self.get_vids_candidate(record.rid, record.vid_next, vids_visited, True)
            else:
                vid_candidates = self.get_vids_candidate(record.rid, record.vid_next, vids_visited, False)
                predicted_scores.append([])
            score_raw = Variable(torch.zeros(1, len(vid_candidates)))
            for i, vid_candidate in enumerate(vid_candidates):
                score_raw[0, i] = self.get_kde_score_all(records_al, id, vid_candidate, emb_t_al)
            predicted_scores[idx] = F.sigmoid(score_raw) if is_train else F.softmax(score_raw)
            id_vids.append(vid_candidates)
            idx += 1
        return predicted_scores, id_vids, id_vids_true

    def get_kde_score_decay(self, records_al, id, vid_cand):
        score_sum = Variable(torch.zeros([1]))
        weight_sum = Variable(torch.zeros([1]))
        t_cur = records_al[id].tid_next
        for id_r, record in enumerate(records_al):
            if id_r == id + 1:
                continue
            score = self.kde(record.vid, vid_cand)
            if score == 0.0:
                continue
            t_pre = records_al[id_r].tid
            t_diff = abs(t_cur % 24 - t_pre % 24)
            if t_diff >= 12:
                t_diff = 24 - t_diff
            weight = Variable(torch.FloatTensor([math.exp(-t_diff)]))
            score_sum += score * weight
            weight_sum += weight
        return score_sum / weight_sum

    def get_kde_score_all(self, records_al, id, vid_cand, emb_t_al):
        emb_t_cur = emb_t_al[id + 1].view(-1, 1)
        score_sum = Variable(torch.zeros([1]))
        weight_sum = Variable(torch.zeros([1]))
        for id_r, record in enumerate(records_al):
            if id_r == id + 1:
                continue
            score = self.kde(record.vid, vid_cand)
            if score == 0.0:
                continue
            emb_t_pre = emb_t_al[id_r].view(1, -1)
            weight = torch.mm(emb_t_pre, emb_t_cur)
            # print 'score: ', score, type(score)
            # print 'weight: ', weight
            score_sum += score * weight
            weight_sum += weight
        return score_sum / weight_sum

    def kde(self, vid_pre, vid):
        bandwidth = self.vid_band[vid_pre]
        coor_diff = self.vid_coor_nor[vid] - self.vid_coor_nor[vid_pre]
        return float(0.159154943 / bandwidth * np.exp(-0.5 * np.sum(coor_diff ** 2) / bandwidth))

    def get_vids_candidate(self, rid, vid_true=None, vids_visited=None, is_train=True):
        if rid in self.rid_sampling_info:
            vids, probs = self.rid_sampling_info[rid]
        else:
            nbs = set()
            for vid_visited in vids_visited:
                vids = self.tree.query_radius([self.vid_coor_nor[vid_visited]], r=self.dist_threshold)
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

def train(root_path, dataset, adapt_bandwidth=True, bandwidth_global=0.07, adapt_nn=5, n_iter=500, iter_start=0):
    torch.manual_seed(0)
    random.seed(0)
    dl = pickle.load(open(root_path + 'dl_' + dataset + '.pk', 'rb'))
    for _, records_u in dl.uid_records.items():
        records_u.summarize()
    model = TimeAwareCF(dl.nv, dl.nt, vid_coor_nor=dl.vid_coor_nor, vid_pop=dl.vid_pop, adapt_bandwidth=adapt_bandwidth, bandwidth_global=bandwidth_global, adapt_nn=adapt_nn)
    if iter_start != 0:
        model.load_state_dict(torch.load(root_path + 'model_tcf_' + str(adapt_bandwidth) + '_' +str(bandwidth_global) + '_' + str(nn) + '_' + str(iter_start) + '.md'))
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
            if idx % 10 == 0:
                print 'uid: \t%d\tloss: %f' % (idx, print_loss_total)
        print iter, print_loss_total
        if iter % 5 == 0:
            torch.save(model.state_dict(), root_path + 'model_tcf_' + str(adapt_bandwidth) + '_' +str(bandwidth_global) + '_' + str(adapt_nn) + '_' + str(iter) + '.md')

def test(root_path, dataset, adapt_bandwidth=True, bandwidth_global=0.07, adapt_nn=5, iter_start=0):
    torch.manual_seed(0)
    random.seed(0)
    dl = pickle.load(open(root_path + 'dl_' + dataset + '.pk', 'rb'))
    model = TimeAwareCF(dl.nv, dl.nt, vid_coor_nor=dl.vid_coor_nor, vid_pop=dl.vid_pop, adapt_bandwidth=adapt_bandwidth, bandwidth_global=bandwidth_global, adapt_nn=adapt_nn)
    for _, records_u in dl.uid_records.items():
        records_u.summarize()
    if iter_start != 0:
        model.load_state_dict(torch.load(root_path + 'model_tcf_' + str(adapt_bandwidth) + '_' +str(bandwidth_global) + '_' + str(adapt_nn) + '_' + str(iter_start) + '.md'))
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
        if (uid + 1) % 100 == 0:
            print (uid + 1), hits / cnt
        hits /= cnt
    print 'hits: ', hits, 'cnt: ', cnt

def test_direct(root_path, dataset, adapt_bandwidth=True, bandwidth_global=0.07, adapt_nn=5, iter_start=0):
    torch.manual_seed(0)
    random.seed(0)
    dl = pickle.load(open(root_path + 'dl_' + dataset + '.pk', 'rb'))
    model = TimeAwareCF(dl.nv, dl.nt, vid_coor_nor=dl.vid_coor_nor, vid_pop=dl.vid_pop, adapt_bandwidth=adapt_bandwidth, bandwidth_global=bandwidth_global, adapt_nn=adapt_nn)
    for _, records_u in dl.uid_records.items():
        records_u.summarize()
    hits = np.zeros(3)
    cnt = 0
    for uid, records_u in dl.uid_records.items():
        id_scores, id_vids, vids_true = model.direct_test(records_u)
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
        if (uid + 1) % 10 == 0:
            print (uid + 1), hits / cnt
        hits /= cnt
    print 'hits: ', hits, 'cnt: ', cnt
# root_path = '/Users/quanyuan/Dropbox/Research/LocationData/small/'
root_path = '/shared/data/qyuan/LocationData/small/'
dataset = 'foursquare'
adapt_nn = 5
# adapt_nn = int(input('please input adapt nn: '))
bandwidth_global = float(input('please input bandwidth global: '))
# train(root_path, dataset, adapt_bandwidth=True, bandwidth_global=0.07, adapt_nn=adapt_nn)
test_direct(root_path, dataset, adapt_bandwidth=False, bandwidth_global=bandwidth_global, adapt_nn=adapt_nn)