import string
import datetime
import numpy as np
import pickle
import math
hour_gap = 6
valid_portion = 0.1
test_portion = 0.2

class DataLoader(object):
    def __init__(self, hour_gap=6):
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
            dt = datetime.datetime.strptime(al[1].strip('"'), '%Y-%m-%dT%H:%M:%SZ')
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

    def write_to_files(self, root_path, dataset):
        f_coor_rad = open(root_path + dataset + "_coor_rad.txt", 'w')
        f_coor = open(root_path + dataset + "_coor.txt", 'w')
        f_train = open(root_path + dataset + "_train.txt", 'w')
        f_test = open(root_path + dataset + "_test.txt", 'w')
        for uid, records_u in self.uid_records.items():
            vids_long = [[], []]
            vids_short = [[], []]
            tids = [[], []]
            vids_next = [[], []]
            tids_next = [[], []]
            for rid, record in enumerate(records_u.records):
                role_id = 0 if rid < records_u.test_idx else 1
                if record.is_first:
                    vids_session = []
                if record.is_last:
                    vids_short[role_id].append(vids_session)
                    vids_session = []
                vids_long[role_id].append(record.vid)
                tids[role_id].append(record.tid)
                vids_next[role_id].append(record.vid_next)
                tids_next[role_id].append(record.tid_next)
            f_train.write(str(uid) + ',' + len(vids_short[0]))
            f_test.write(str(uid) + ',' + len(vids_short[1]))
            f_train.write(','.join([str(vid) for vid in vids_long[0]]) + '\n')
            f_test.write(','.join([str(vid) for vid in vids_long[1]]) + '\n')
            for vids_session in vids_short[0]:
                f_train.write(','.join([str(vid) for vid in vids_session]) + '\n')
            for vids_session in vids_short[1]:
                f_test.write(','.join([str(vid) for vid in vids_session]) + '\n')
            f_train.write(','.join([str(tid) for tid in tids[0]]) + '\n')
            f_test.write(','.join([str(tid) for tid in tids[1]]) + '\n')
            f_train.write(','.join([str(vid) for vid in vids_next[0]]) + '\n')
            f_test.write(','.join([str(vid) for vid in vids_next[1]]) + '\n')
            f_train.write(','.join([str(tid) for tid in tids_next[0]]) + '\n')
            f_test.write(','.join([str(tid) for tid in tids_next[1]]) + '\n')
        for vid in range(self.nv):
            f_coor.write(','.join([str(coor) for coor in self.vid_coor[vid]]) + '\n')
            f_coor_rad.write(','.join([str(coor) for coor in self.vid_coor_rad[vid]]) + '\n')
        f_train.close()
        f_test.close()
        f_coor.close()
        f_coor_rad.close()
        f_u = open(root_path + dataset + "_u.txt", 'w')
        f_v = open(root_path + dataset + "_v.txt", 'w')
        for u in self.u_uid:
            f_u.write(u + ',' + str(self.u_uid[u]) + '\n')
        for v in self.v_vid:
            f_v.write(v + ',' + str(self.v_vid[v]) + '\n')
        f_u.close()
        f_v.close()

class Record(object):
    def __init__(self, dt, uid, vid, vid_next=-1, tid_next = -1, is_first=False, is_last=False, rid=None):
        self.dt = dt
        self.rid = rid
        self.uid = uid
        self.vid = vid
        self.tid = dt.hour
        if dt.weekday > 4:
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

# class UserRecords(object):
#     def __init__(self, uid):
#         self.uid = uid
#         self.records = []
#         self.session_cnt = 0
#         self.session_start_idx=[]
#         self.dt_last = None
#         self.test_idx = 0
#         self.record_cnt_train = 0
#         self.record_cnt_test = 0
#
#     def add_record(self, dt, uid, vid):
#         is_first = False
#         if self.dt_last is None or (dt - self.dt_last).total_seconds() / 3600.0 > hour_gap:
#             self.session_start_idx.append(len(self.records))
#             self.session_cnt += 1
#             self.test_idx = self.session_start_idx[int(self.session_cnt * (1 - test_portion))]
#             is_first = True
#             if len(self.records) > 0:
#                 self.records[len(self.records) - 1].is_last = True
#         record = Record(dt, uid, vid, is_first=is_first, is_last=True)
#         if len(self.records) > 0:
#             self.records[len(self.records) - 1].vid_next = record.vid
#             self.records[len(self.records) - 1].tid_next = record.tid
#             if not is_first:
#                 self.records[len(self.records) - 1].is_last = False
#             else:
#                 self.records[len(self.records) - 1].vid_next = -1
#         self.records.append(record)
#         self.dt_last = dt
#
#     def get_records(self, mod=0):
#         if mod == 0:  # train only
#             return self.records[0: self.test_idx]
#         elif mod == 1:  # test only
#             return self.records[self.test_idx: len(self.records)]
#         else:
#             return self.records
#
#     def get_predicting_records_cnt(self, mod=0):
#         cnt = 0
#         if mod == 0:  # train only
#             for record in self.records[0: self.test_idx]:
#                 if record.is_last:
#                     continue
#                 cnt += 1
#             return cnt
#         else:  # test only
#             for record in self.records[self.test_idx: len(self.records)]:
#                 if record.is_last:
#                     continue
#                 cnt += 1
#             return cnt
