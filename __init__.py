import torch
import os
import pickle
import attention_model_enhance
import SimpleModelDecoder
import string
import datetime
import numpy as np
import NCF
import JNTM
import SERM
import DSSM


hour_gap = 6
valid_portion = 0.1
test_portion = 0.2
user_session_min = 3 #!!! previous 3
session_len_min = 2
session_len_4sq_max = 20
session_len_gb_max = 40
def analyze_time_dist(root_path, dataset):
    dl = pickle.load(open(root_path + 'dl_' + dataset + '.pk', 'rb'))
    tid_cnt = {}
    for _, records_u in dl.uid_records.items():
        for record in records_u.records:
            tid = record.tid % 24
            if tid not in tid_cnt:
                tid_cnt[tid] = 0
            tid_cnt[tid] += 1
    for tid in sorted(tid_cnt.keys()):
        print tid, tid_cnt[tid]
    raw_input()

def analyze_session_len(root_path, dataset):
    dl = pickle.load(open(root_path + 'dl_' + dataset + '.pk', 'rb'))
    len_cnt = {}
    for _, records_u in dl.uid_records.items():
        for record in records_u.records:
            if record.is_first:
                len = 0
            len += 1
            if record.is_last:
                if len not in len_cnt:
                    len_cnt[len] = 0
                len_cnt[len] += 1
    for len in len_cnt:
        print len, len_cnt[len]
    raw_input()

class DataLoader(object):
    def __init__(self, hour_gap=6, offset=0):
        self.hour_gap = hour_gap
        self.u_uid = {}
        self.uid_u = {}
        self.v_vid = {}
        self.uid_records = {}
        self.nu = 0
        self.nv = 0
        self.nt = 24 * 2
        self.nr = 0
        self.vid_coor = {}
        self.vid_coor_rad = {}
        self.vid_coor_nor = {}
        self.vid_coor_nor_rectified = {}
        self.vid_pop = {}
        self.sampling_list = []
        self.offset = offset

    def summarize(self):
        for uid, record_u in self.uid_records.items():
            record_u.summarize()

    def add_records(self, file_path, dl_save_path, u_cnt_max=-1, blacklist=None):
        f = open(file_path, 'r', -1)
        for line in f:
            al = line.strip().split('\t')
            u = al[0]
            if blacklist is not None and u in blacklist:
                continue
            v = al[4]
            dt = datetime.datetime.strptime(al[1].strip('"'), '%Y-%m-%dT%H:%M:%SZ') - datetime.timedelta(minutes=self.offset)
            start_2009 = datetime.datetime.strptime('2009-03-08T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
            end_2009 = datetime.datetime.strptime('2009-11-01T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
            start_2010 = datetime.datetime.strptime('2010-03-14T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
            end_2010 = datetime.datetime.strptime('2010-11-07T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
            start_2011 = datetime.datetime.strptime('2011-03-13T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
            end_2011 = datetime.datetime.strptime('2011-11-06T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
            start_2012 = datetime.datetime.strptime('2012-03-11T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
            end_2012 = datetime.datetime.strptime('2012-11-04T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
            year = dt.year
            if year == 2009 and dt > start_2009 and dt < end_2009 or \
                year == 2010 and dt > start_2010 and dt < end_2010 or \
                year == 2011 and dt > start_2011 and dt < end_2011 or \
                year == 2012 and dt > start_2012 and dt < end_2012:
                dt = dt + datetime.timedelta(minutes=60)

            lat = string.atof(al[2])
            lng = string.atof(al[3])
            if u not in self.u_uid:
                if u_cnt_max > 0 and len(self.u_uid) >= u_cnt_max:
                    break
                # print u, self.nu
                self.u_uid[u] = self.nu
                self.uid_u[self.nu] = u
                self.uid_records[self.nu] = UserRecords(self.nu)
                self.nu += 1
            if v not in self.v_vid:
                self.v_vid[v] = self.nv
                self.vid_pop[self.nv] = 0
                self.vid_coor_rad[self.nv] = np.array([np.radians(lat), np.radians(lng)])
                self.vid_coor[self.nv] = np.array([lat, lng])
                self.nv += 1
            uid = self.u_uid[u]
            vid = self.v_vid[v]
            self.sampling_list.append(vid)
            self.vid_pop[vid] += 1
            self.uid_records[uid].add_record(dt, uid, vid, self.nr)
            self.nr += 1
        f.close()

        coor_mean = np.zeros(2)
        coor_var = np.zeros(2)
        for vid, coor in self.vid_coor.items():
            coor_mean += coor
        coor_mean /= len(self.vid_coor)
        for vid, coor in self.vid_coor.items():
            coor_var += (coor - coor_mean) ** 2
        coor_var /= len(self.vid_coor)
        coor_var = np.sqrt(coor_var)
        for vid in self.vid_coor:
            self.vid_coor_nor[vid] = (self.vid_coor[vid] - coor_mean) / coor_var
            lat_sub = self.vid_coor[vid][0] - coor_mean[0]
            lng_sub = self.vid_coor[vid][1] - coor_mean[1]
            lat_rectified = lat_sub / coor_var[0]
            lng_rectified = lng_sub * math.cos(self.vid_coor[vid][0]) / coor_var[0]
            self.vid_coor_nor_rectified[vid] = np.array([lat_rectified, lng_rectified])
        if blacklist is not None:
            pickle.dump(self, open(dl_save_path, 'wb'))

    def show_info(self):
        print 'U: ', self.nu, 'V: ', self.nv, 'R: ', self.nr, 'T: ', self.nt

    def write_to_files(self, root_path):
        self.summarize()
        # f_coor_nor = open(root_path + "coor_nor.txt", 'w')
        f_train = open(root_path + "train.txt", 'w')
        f_test = open(root_path + "test.txt", 'w')
        for uid, records_u in self.uid_records.items():
            vids_long = [[], []]
            vids_short_al = [[], []]
            tids = [[], []]
            vids_next = [[], []]
            tids_next = [[], []]
            for rid, record in enumerate(records_u.records):
                if record.is_first:
                    vids_short = []
                vids_short.append(record.vid)
                if rid < records_u.test_idx:
                    vids_long[0].append(record.vid)
                    tids[0].append(record.tid)
                    vids_next[0].append(record.vid_next)
                    tids_next[0].append(record.tid_next)
                vids_long[1].append(record.vid)
                tids[1].append(record.tid)
                vids_next[1].append(record.vid_next)
                tids_next[1].append(record.tid_next)
                if record.is_last:
                    if rid < records_u.test_idx:
                        vids_short_al[0].append(vids_short)
                    vids_short_al[1].append(vids_short)
                    vids_short = []
                #
                #
                #
                #
                # role_id = 0 if rid < records_u.test_idx else 1
                # if record.is_first:
                #     vids_short = []
                # vids_long[role_id].append(record.vid)
                # vids_short.append(record.vid)
                # tids[role_id].append(record.tid)
                # vids_next[role_id].append(record.vid_next)
                # tids_next[role_id].append(record.tid_next)
                # if record.is_last:
                #     vids_short_al[role_id].append(vids_short)
                #     vids_short = []
            f_train.write(str(uid) + ',' + str(len(vids_short_al[0])) + ',' + str(records_u.test_idx) + '\n')
            f_test.write(str(uid) + ',' + str(len(vids_short_al[1])) + ',' + str(records_u.test_idx) + '\n')
            f_train.write(','.join([str(vid) for vid in vids_long[0]]) + '\n')
            f_test.write(','.join([str(vid) for vid in vids_long[1]]) + '\n')
            for vids_short in vids_short_al[0]:
                f_train.write(','.join([str(vid) for vid in vids_short]) + '\n')
            for vids_short in vids_short_al[1]:
                f_test.write(','.join([str(vid) for vid in vids_short]) + '\n')
            f_train.write(','.join([str(tid) for tid in tids[0]]) + '\n')
            f_test.write(','.join([str(tid) for tid in tids[1]]) + '\n')
            f_train.write(','.join([str(vid) for vid in vids_next[0]]) + '\n')
            f_test.write(','.join([str(vid) for vid in vids_next[1]]) + '\n')
            f_train.write(','.join([str(tid) for tid in tids_next[0]]) + '\n')
            f_test.write(','.join([str(tid) for tid in tids_next[1]]) + '\n')
        coor_nor = np.zeros((len(self.vid_coor_nor), 2), dtype=np.float64)
        for vid in range(self.nv):
            coor_nor[vid] = self.vid_coor_nor[vid]
        np.savetxt(root_path + 'coor_nor.txt', coor_nor, fmt="%lf", delimiter=',')
        # coor_nor.tofile(root_path + 'coor_nor.txt', sep=',')
            # f_coor_nor.write(','.join([str(coor) for coor in self.vid_coor_nor[vid]]) + '\n')
        f_train.close()
        f_test.close()
        # f_coor_nor.close()
        f_u = open(root_path + "u.txt", 'w')
        f_v = open(root_path + "v.txt", 'w')
        f_t = open(root_path + "t.txt", 'w')
        for u in self.u_uid:
            f_u.write(u + ',' + str(self.u_uid[u]) + '\n')
        for v in self.v_vid:
            f_v.write(v + ',' + str(self.v_vid[v]) + '\n')
        for t in xrange(48):
            f_t.write(str(t) + ',' + str(t) + '\n')
        f_u.close()
        f_v.close()
        f_t.close()

class Record(object):
    def __init__(self, dt, uid, vid, vid_next=-1, tid_next = -1, is_first=False, is_last=False, rid=None):
        self.dt = dt
        self.rid = rid
        self.uid = uid
        self.vid = vid
        self.tid = dt.hour
        if dt.weekday > 4 or dt.weekday == 4 and dt.hour>=18:
            self.tid += 24
        self.tid_168 = dt.weekday() * 24 + dt.hour
        self.vid_next = vid_next
        self.tid_next = tid_next
        self.is_first = is_first
        self.is_last = is_last

    def peek(self):
        print 'u: ', self.uid, '\tv: ', self.vid, '\tt: ', self.tid, '\tvid_next: ', self.vid_next, '\tis_first: ', self.is_first, '\tis_last: ', self.is_last, 'dt: ', self.dt, 'rid: ', self.rid

class UserRecords(object):
    def __init__(self, uid):
        self.uid = uid
        self.records = []
        self.dt_last = None
        self.test_idx = 0

    def add_record(self, dt, uid, vid, rid=None):
        is_first = False
        if self.dt_last is None or (dt - self.dt_last).total_seconds() / 3600.0 > hour_gap:
            is_first = True
            if len(self.records) > 0:
                self.records[len(self.records) - 1].is_last = True
        record = Record(dt, uid, vid, is_first=is_first, is_last=True, rid=rid)
        if len(self.records) > 0:
            self.records[len(self.records) - 1].vid_next = record.vid
            self.records[len(self.records) - 1].tid_next = record.tid
            if not is_first:
                self.records[len(self.records) - 1].is_last = False
            else:
                self.records[len(self.records) - 1].vid_next = -1
        self.records.append(record)
        self.dt_last = dt
        self.is_valid = True

    def summarize(self):
        session_begin_idxs = []
        session_len = 0
        session_begin_idx = 0
        for rid, record in enumerate(self.records):
            if record.is_first:
                session_begin_idx = rid
            session_len += 1
            if record.is_last:
                if session_len >= 2:
                    session_begin_idxs.append(session_begin_idx)
                session_len = 0
        if len(session_begin_idxs) < 2:
            self.is_valid = False
            return
        test_session_idx = int(len(session_begin_idxs) * (1 - test_portion))
        if test_session_idx == 0:
            test_session_idx = 1
        if test_session_idx < len(session_begin_idxs):
            self.test_idx = session_begin_idxs[test_session_idx]
        else:
            self.is_valid = False


    def valid(self):
        return self.is_valid

    def get_records(self, mod=0):
        if mod == 0:  # train only
            return self.records[0: self.test_idx]
        elif mod == 1:  # test only
            return self.records[self.test_idx: len(self.records)]
        else:
            return self.records

    def get_predicting_records_cnt(self, mod=0):
        cnt = 0
        if mod == 0:  # train only
            for record in self.records[0: self.test_idx]:
                if record.is_last:
                    continue
                cnt += 1
            return cnt
        else:  # test only
            for record in self.records[self.test_idx: len(self.records)]:
                if record.is_last:
                    continue
                cnt += 1
            return cnt

if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(0)
    root_path = '/Users/quanyuan/Dropbox/Research/LocationData/' \
        if os.path.exists('/Users/quanyuan/Dropbox/Research/LocationData/') \
        else 'LocationData/'
    small_path = root_path + 'small/'
    dataset = 'foursquare'
    # analyze_time_dist(small_path, 'gowalla')
    dl = pickle.load(open(small_path + 'dl_' + dataset + '.pk', 'rb'))
    '''
    task = int(input('please input task (0: train, 1: test, 2: baselines): '))
    model = int(input('please input model (0: our, 1: decoder): '))
    mod = int(input("input mod: "))
    iter = int(input('please input last iter: '))
    '''
    task = 0
    model = 0
    mod = 2
    iter = 0
    print "1:JNTM 2:SERM 3:NCF 4:DSSM"
    temp = int(input('which baseline'))
    if task == 0:
        if model == 0:
            #attention_model_enhance.train(dl,small_path, dataset, iter_start=iter, mod=mod)
            if temp == 1:
                #JNTM.train(dl,3555,500)
                JNTM.train(dl,small_path)
            elif temp == 2:
                SERM.train(dl,small_path)
            elif temp == 3:
                NCF.train(dl,small_path,dataset)
            elif temp ==4:
                DSSM.train(dl,small_path,dataset)

        elif model == 1:
            SimpleModelDecoder.train(dl,small_path, dataset, iter_start=iter, mod=mod)
    else:
        if model == 0:
            attention_model_enhance.test(dl,small_path, dataset, iter_start=iter, mod=mod)
        elif model == 1:
            SimpleModelDecoder.test(dl,small_path, dataset, iter_start=iter, mod=mod)
