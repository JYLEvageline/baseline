import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import string
import random
import pickle
import os
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import torch.optim as optim

def euclidean_analysis_rank_vs_dist(root_path, dataset):
    dl = pickle.load(open(root_path + 'dl_' + dataset + '.pk', 'rb'))
    coor_nor = [dl.vid_coor_nor[vid] for vid in range(dl.nv)]
    tree = KDTree(coor_nor)
    cnt_avg_dist = 0
    cnt_avg_rank = 0
    cnt = 0
    for uid, records_u in dl.uid_records.items():
        vid_cnt = {}
        records_u.summarize()
        for record in records_u.get_records(0):
            if record.vid not in vid_cnt:
                vid_cnt[record.vid] = 0
            vid_cnt[record.vid] += 1
        records_al_test = records_u.get_records(1)
        for rid, record in enumerate(records_al_test):
            if record.is_last:
                continue
            if record.vid not in vid_cnt:
                vid_cnt[record.vid] = 0
            vid_cnt[record.vid] += 1
            vid_candidates_dist = set()
            vid_candidates_rank = set()
            for vid in vid_cnt:
                ids = tree.query_radius([dl.vid_coor_nor[vid]], r=0.07)
                for vid_candidate in ids[0]:
                    vid_candidates_dist.add(vid_candidate)
                _, ids = tree.query([dl.vid_coor_nor[vid]], k=35)
                for vid_candidate in ids[0]:
                    vid_candidates_rank.add(vid_candidate)
            cnt_avg_dist += len(vid_candidates_dist)
            cnt_avg_rank += len(vid_candidates_rank)
            cnt += 1
    cnt_avg_dist /= cnt
    cnt_avg_rank /= cnt
    print 'cnt_avg_dist: ', cnt_avg_dist
    print 'cnt_avg_rank: ', cnt_avg_rank

def euclidean_analysis_dist_all(root_path, dataset):
    dl = pickle.load(open(root_path + 'dl_' + dataset + '.pk', 'rb'))
    D = 1
    D_bin = 0.01
    dl.show_info()
    coor_nor = [dl.vid_coor_nor[vid] for vid in range(dl.nv)]
    time_dist_all = np.zeros((12, int(D / D_bin)))
    dist_all = np.zeros(int(D / D_bin))
    time_cnt = np.zeros(12)
    for uid, records_u in dl.uid_records.items():
        vid_cnt = {}
        records_u.summarize()
        for record in records_u.get_records(0):
            if record.vid not in vid_cnt:
                vid_cnt[record.vid] = 0
            vid_cnt[record.vid] += 1
        records_al_test = records_u.get_records(1)
        for rid, record in enumerate(records_al_test):
            if record.is_last:
                continue
            if record.vid not in vid_cnt:
                vid_cnt[record.vid] = 0
            vid_cnt[record.vid] += 1

            time_gap = int((records_al_test[rid + 1].dt - records_al_test[rid].dt).total_seconds() / 60 / 30)
            if time_gap >= 12:
                time_gap = 11
            time_cnt[time_gap] += 1
            min_dist = 10000
            for vid in vid_cnt:
                dist = np.sqrt(np.sum((dl.vid_coor_nor[record.vid_next] - dl.vid_coor_nor[vid]) ** 2))
                if dist < min_dist:
                    min_dist = dist
            # print 'min_dist: ', min_dist
            idx = int(min_dist / D_bin)
            # print 'idx: ', idx
            if idx >= int(D / D_bin):
                idx = int(D / D_bin) - 1
            # raw_input()
            # print 'idx: ', idx
            time_dist_all[time_gap, idx] += 1
            dist_all[idx] += 1
    for i in xrange(0, len(time_cnt)):
        time_dist_all[i] /= time_cnt[i]
    dist_all /= np.sum(dist_all)
    plt.imshow(time_dist_all, cmap='hot', interpolation='nearest')
    plt.show()
    for i in xrange(0, len(time_cnt)):
        for j in xrange(1, len(time_dist_all[i])):
            time_dist_all[i, j] += time_dist_all[i, j - 1]
        print i, time_dist_all[i]
    print 0, dist_all[0]
    for j in xrange(1, len(dist_all)):
        dist_all[j] += dist_all[j - 1]
        print j * D_bin, dist_all[j]


def euclidean_analysis_rank_all(root_path, dataset):
    dl = pickle.load(open(root_path + 'dl_' + dataset + '.pk', 'rb'))
    K = min(5000, dl.nv)
    K_bin = 10
    dl.show_info()
    coor_nor = [dl.vid_coor_nor[vid] for vid in range(dl.nv)]
    tree = KDTree(coor_nor)
    time_rank_all = np.zeros((12, K / K_bin))
    rank_all = np.zeros(K / K_bin)
    time_cnt = np.zeros(12)
    for uid, records_u in dl.uid_records.items():
        vid_cnt = {}
        #all visable
        records_u.summarize()
        for record in records_u.get_records(0):
            if record.vid not in vid_cnt:
                vid_cnt[record.vid] = 0
            vid_cnt[record.vid] += 1
        # print 'vid_cnt: ', vid_cnt
        records_al_test = records_u.get_records(1)
        for rid, record in enumerate(records_al_test):
            if record.is_last:
                continue
            if record.vid not in vid_cnt:
                vid_cnt[record.vid] = 0
            vid_cnt[record.vid] += 1

            time_gap = int((records_al_test[rid + 1].dt - records_al_test[rid].dt).total_seconds() / 60 / 30)
            if time_gap >= 12:
                time_gap = 11
            time_cnt[time_gap] += 1
            min_rank = min(5000, dl.nv) / K_bin - 1
            for vid in vid_cnt:
                dist = np.sqrt(np.sum((dl.vid_coor_nor[record.vid_next] - dl.vid_coor_nor[vid]) ** 2))
                ids = tree.query_radius([dl.vid_coor_nor[vid]], r=dist)
                rank = len(ids[0]) / K_bin
                if rank < min_rank:
                    min_rank = rank
            time_rank_all[time_gap, min_rank] += 1
            rank_all[min_rank] += 1
    for i in xrange(0, len(time_cnt)):
        time_rank_all[i] /= time_cnt[i]
    rank_all /= np.sum(rank_all)
    plt.imshow(time_rank_all, cmap='hot', interpolation='nearest')
    plt.show()
    for i in xrange(0, len(time_cnt)):
        for j in xrange(1, len(time_rank_all[i])):
            time_rank_all[i, j] += time_rank_all[i, j - 1]
        print i, time_rank_all[i]
    print 0, rank_all[0]
    for j in xrange(1, len(rank_all)):
        rank_all[j] += rank_all[j - 1]
        print j * K_bin, rank_all[j]

def euclidean_analysis_rank_pre(root_path, dataset):
    dl = pickle.load(open(root_path + 'dl_' + dataset + '.pk', 'rb'))
    K = min(5000, dl.nv)
    K_bin = 10
    dl.show_info()
    coor_nor = [dl.vid_coor_nor[vid] for vid in range(dl.nv)]
    tree = KDTree(coor_nor)
    time_rank_pre = np.zeros((12, K / K_bin))
    rank_pre = np.zeros(K / K_bin)
    time_cnt = np.zeros(12)
    for uid, records_u in dl.uid_records.items():
        vid_cnt = {}
        # all visable
        records_u.summarize()
        for record in records_u.get_records(0):
            if record.vid not in vid_cnt:
                vid_cnt[record.vid] = 0
            vid_cnt[record.vid] += 1
        # print 'vid_cnt: ', vid_cnt
        records_al_test = records_u.get_records(1)
        for rid, record in enumerate(records_al_test):
            if record.is_last:
                continue
            if record.vid not in vid_cnt:
                vid_cnt[record.vid] = 0
            vid_cnt[record.vid] += 1
            time_gap = int((records_al_test[rid + 1].dt - records_al_test[rid].dt).total_seconds() / 60 / 30)
            if time_gap == 12:
                record.peek()
                # records_u.records[rid + 1].peek()
                # print (records_u.records[rid + 1].dt - records_u.records[rid].dt).total_seconds() / 60 / 30
                # raw_input()
                time_gap = 11
            time_cnt[time_gap] += 1
            dist = np.sqrt(np.sum((dl.vid_coor_nor[record.vid] - dl.vid_coor_nor[record.vid_next]) ** 2))
            ids = tree.query_radius([dl.vid_coor_nor[record.vid]], r=dist)
            idx = len(ids[0]) / K_bin
            if idx >= K / K_bin:
                idx = K / K_bin - 1
            time_rank_pre[time_gap, idx] += 1
            rank_pre[idx] += 1
    for i in xrange(0, len(time_cnt)):
        time_rank_pre[i] /= time_cnt[i]
    rank_pre /= np.sum(rank_pre)
    plt.imshow(time_rank_pre, cmap='hot', interpolation='nearest')
    plt.show()
    for i in xrange(0, len(time_cnt)):
        for j in xrange(1, len(time_rank_pre[i])):
            time_rank_pre[i, j] += time_rank_pre[i, j - 1]
        print i, time_rank_pre[i]
    print 0, rank_pre[0]
    for j in xrange(1, len(rank_pre)):
        rank_pre[j] += rank_pre[j - 1]
        print j * 100, rank_pre[j]



    # calculate rank of groundtruth to all visited vid

# def time_recent_history_distance(root_path):
#     K = 6   #neighbor
#     D = 50  #MAX Dis bin
#     dl = pickle.load(open(root_path + 'dl.pk', 'rb'))
#     dl.show_info()
#     dist_map_pre = np.zeros((12, D), dtype=np.float64)
#     dist_map_avg = np.zeros((12, D), dtype=np.float64)
#     dist_map_div = np.zeros((12, D), dtype=np.float64)
#     time_cnt = np.zeros(12)
#     time_cnt_div = np.zeros(12)
#     for uid, records_u in dl.uid_records.items():   # for user in users
#         # print 'user: ', uid
#         id_vid = []
#         id_coor_nor = []
#         vid_set = set()
#         for record in records_u.records:
#             if record.is_last:
#                 continue
#             if record.is_first:
#                 if record.vid not in vid_set:
#                     id_vid.append(record.vid)
#                     id_coor_nor.append(dl.vid_coor_nor[record.vid])
#                     # vid_set.add(record.vid)
#             if record.vid_next not in vid_set:
#                 id_vid.append(record.vid_next)
#                 id_coor_nor.append(dl.vid_coor_nor[record.vid_next])
#                 # vid_set.add(record.vid_next)
#         tree = KDTree(id_coor_nor)
#         dist_metric = DistanceMetric.get_metric('euclidean')
#
#         for rid, record in enumerate(records_u.records):
#             if record.is_last:
#                 continue
#             time_gap = int((records_u.records[rid + 1].dt - records_u.records[rid].dt).total_seconds() / 60 / 30)      #half hour
#             dist_pre = dist_metric.pairwise([dl.vid_coor_rad[record.vid_next], dl.vid_coor_rad[record.vid]])[0][1]
#             dist, ids = tree.query([dl.vid_coor_rad[record.vid_next]], min(len(id_vid), K))
#             dist_avg = 0
#             skipped = False
#             cnt = 0
#             for idx in range(min(len(id_vid), K)):
#                 vid = id_vid[ids[0, idx]]
#                 if vid == record.vid_next and not skipped:
#                     skipped = True
#                     continue
#                 dist_avg += dist[0, idx]
#                 cnt += 1
#             if cnt == 0:
#                 continue
#             dist_avg /= cnt
#             # print dist_avg, dist_pre, time_gap, dist_avg / dist_pre
#             if dist_pre > 0:
#                 dist_div = dist_avg / dist_pre
#                 dist_div = int(dist_div * 10)
#                 if dist_div < D:
#                     dist_map_div[time_gap, dist_div] += 1
#                     time_cnt_div[time_gap] += 1
#             dist_avg = int(dist_avg)
#             dist_pre = int(dist_pre)
#             if dist_avg >= D or dist_pre >= D:
#                 continue
#             if dist_avg >= D:
#                 dist_avg = D - 1
#             if dist_pre >= D:
#                 dist_pre = D - 1
#             dist_map_pre[time_gap, dist_pre] += 1
#             dist_map_avg[time_gap, dist_avg] += 1
#             time_cnt[time_gap] += 1
#     for i in range(0, len(time_cnt)):
#         dist_map_avg[i] /= time_cnt[i]
#         dist_map_pre[i] /= time_cnt[i]
#         if time_cnt_div[i] > 0:
#             dist_map_div[i] /= time_cnt_div[i]
#         # for j in range(1, len(dist_map_pre[i])):
#         #     if dist_map_pre[i, j] == 0:
#         #         continue
#         #     dist_map_div[i, j] = dist_map_avg[i, j] / dist_map_pre[i, j]
#     print "dist_map_pre: ", dist_map_pre
#     print "dist_map_avg: ", dist_map_avg
#     print "dist_map_div: ", dist_map_div
#     plt.imshow(dist_map_pre, cmap='hot', interpolation='nearest')
#     plt.show()
#     plt.hold(True)
#     plt.imshow(dist_map_avg, cmap='hot', interpolation='nearest')
#     plt.show()
#     plt.hold(True)
#     plt.imshow(dist_map_div, cmap='hot', interpolation='nearest')
#     plt.show()

# def nearest_distance(root_path):
#     dist_metric = DistanceMetric.get_metric('haversine')
#     dl = pickle.load(open(root_path + 'dl.pk', 'rb'))
#     dl.show_info()
#     dist_bin_nearest = np.zeros(200)
#     dist_bin_pre = np.zeros(200)
#     for uid, records_u in dl.uid_records.items():
#         vid_visited = []
#         for i, record in enumerate(records_u.records):
#             if i == 0:
#                 vid_last = record.vid
#                 vid_visited.append(record.vid)
#                 continue
#             dist_min = 100000
#             for vid_pre in vid_visited:
#                 dist = int(dist_metric.pairwise([dl.vid_coor_rad[vid_pre], dl.vid_coor_rad[record.vid]])[0][1] * 6371 * 10)
#                 dist_min = min(dist, dist_min)
#             dist_pre = int(dist_metric.pairwise([dl.vid_coor_rad[vid_last], dl.vid_coor_rad[record.vid]])[0][1] * 6371 * 10)
#             if dist_min >= 200:
#                 dist_min = 199
#             if dist_pre >= 200:
#                 dist_pre = 199
#             dist_bin_nearest[dist_min] += 1
#             dist_bin_pre[dist_pre] += 1
#             vid_visited.append(record.vid)
#             vid_last = record.vid
#     dist_bin_nearest /= dist_bin_nearest.sum()
#     dist_bin_pre /= dist_bin_pre.sum()
#     for j in range(1, 200):
#         dist_bin_nearest[j] += dist_bin_nearest[j - 1]
#         dist_bin_pre[j] += dist_bin_pre[j - 1]
#         print j * 100, dist_bin_nearest[j], dist_bin_pre[j]


# def time_distance(root_path):
#     dist_metric = DistanceMetric.get_metric('haversine')
#     dl = pickle.load(open(root_path + 'dl.pk', 'rb'))
#     dl.show_info()
#     raw_input()
#     tree = BallTree(dl.vid_coor_rad.values(), leaf_size=40, metric='haversine')
#     coor_pre = None
#     dt_pre = None
#     dis_map = np.zeros((12, 20), dtype=np.float64)
#     nn_map = np.zeros((12, 20), dtype=np.float64)
#     time_cnt = np.zeros(12)
#     dis_max = 0
#     for uid, records_u in dl.uid_records.items():
#         for record in records_u.records:
#             if record.is_first:
#                 coor_pre = dl.vid_coor_rad[record.vid]
#                 dt_pre = record.dt
#                 continue
#             coor_cur = dl.vid_coor_rad[record.vid]
#             dt_cur = record.dt
#             time_gap = int((dt_cur - dt_pre).total_seconds() / 60 / 30)
#             dis_gap = int(dist_metric.pairwise([coor_cur, coor_pre])[0][1] * 6371)
#             dist, ids = tree.query([coor_pre], 1000)
#
#             dt_pre = dt_cur
#             coor_pre = coor_cur
#             if dis_gap >= 20:
#                 continue
#
#             for idx, vid in enumerate(ids[0]):
#                 if vid == record.vid:
#                     nn_map[time_gap, idx / 50] += 1
#                     break
#             time_cnt[time_gap] += 1
#             dis_map[time_gap, dis_gap] += 1
#     for i in range(0, len(time_cnt)):
#         dis_map[i] /= time_cnt[i]
#         nn_map[i] /= time_cnt[i]
#         for j in range(1, len(nn_map[i])):
#             nn_map[i, j] += nn_map[i, j - 1]
#     print nn_map
#     print dis_map
#
#     # print dis_max
#     plt.imshow(dis_map, cmap='hot', interpolation='nearest')
#     plt.show()
#     plt.hold(True)
#     plt.imshow(nn_map, cmap='hot', interpolation='nearest')
#     plt.show()

root_path = '/Users/quanyuan/Dropbox/Research/LocationData/' \
        if os.path.exists('/Users/quanyuan/Dropbox/Research/LocationData/') \
        else '/shared/data/qyuan/'
# small_path = root_path + 'full/'
small_path = root_path + 'small/'
dataset = 'foursquare'
# euclidean_analysis_rank_pre(small_path, dataset)
# euclidean_analysis_rank_all(small_path, dataset)
# euclidean_analysis_dist_all(small_path, dataset)
euclidean_analysis_rank_vs_dist(small_path, dataset)
