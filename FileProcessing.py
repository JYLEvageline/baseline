import datetime
import string
import os
import threading
import re
def generate_session_file(file_path, save_path, hour_gap=6):
    fr = open(file_path, 'rt', -1)
    fw = open(save_path, 'wt')
    u_session = []
    session = []
    u_pre = 'start'
    dt_pre = None
    u_set = set()
    for line in fr:
        al = line.split('\t')
        u = al[0]
        u_set.add(u)
        dt = datetime.datetime.strptime(al[1].strip('"'), '%Y-%m-%dT%H:%M:%SZ')
        if u != u_pre:
            # print u_pre, len(u_session)
            if len(u_session) >= 3:
                session_valid_cnt = 0
                for session in u_session:
                    if len(session) >= 2:
                        session_valid_cnt += 1
                if session_valid_cnt >= 3:
                    for session in u_session:
                        for l in session:
                            fw.write(l + '\n')
            u_session = []
            session = []
            dt_pre = None
        if dt_pre is None:
            dt_pre = dt
        if (dt - dt_pre).total_seconds() / 3600.0 > hour_gap:
            # if len(session) >= 2:
            u_session.append(session)
            session = []
        session.append(line.strip())
        u_pre = u
        dt_pre = dt
    u_session.append(session)
    if len(u_session) >= 3:
        session_valid_cnt = 0
        for session in u_session:
            if len(session) >= 2:
                session_valid_cnt += 1
        if session_valid_cnt >= 3:
            for session in u_session:
                for l in session:
                    fw.write(l + '\n')
    fr.close()
    fw.close()

def process_raw_file(file_path, save_path, u_freq_min=10, v_freq_min = 10, u_vcnt_min=0, lat_min=-180, lat_max=180, lng_min=-180, lng_max=180):
    u_freq = {}
    v_freq = {}
    f = open(file_path, 'r', -1)
    for line in f:
        al = line.strip().split('\t')
        u = al[0]
        u_freq[u] = 1 if u not in u_freq else u_freq[u] + 1
    f.close()
    f = open(file_path, 'r', -1)
    for line in f:
        al = line.strip().split('\t')
        u = al[0]
        v = al[4]
        if u_freq[u] < u_freq_min:
            continue
        v_freq[v] = 1 if v not in v_freq else v_freq[v] + 1
    f.close()
    u_set = set()
    fr = open(file_path, 'rt')
    fw = open(save_path, 'wt')
    lines = []
    v_set = set()
    u_pre = '0'
    for line in fr:
        al = line.strip().split('\t')
        u = al[0]
        v = al[4]
        lat = string.atof(al[2])
        lng = string.atof(al[3])
        if lat < lat_min or lat > lat_max or lng < lng_min or lng > lng_max:
            continue
        if u_freq[u] < u_freq_min or v_freq[v] < v_freq_min:
            continue
        v_set.add(v)
        if u == u_pre:
            lines.append(line)
        else:
            u_set.add(u)
            for l in lines[::-1] if file_path.find('foursquare') == -1 else lines:
                fw.write(l)
            lines = [line]
            u_pre = u
    u_set.add(u)
    for l in lines[::-1] if file_path.find('foursquare') == -1 else lines:
        fw.write(l)
    fr.close()
    fw.close()
    print 'U: ', len(u_set)
    print 'V: ', len(v_set)


def read_checkins(checkins_path, venue_path, save_path):
    # read venue coordinates
    f = open(venue_path, 'r')
    venue_lat = {}
    venue_lng = {}
    for line in f:
        al = line.strip().split(",")
        venue = al[0]
        lat = string.atof(al[1])
        lng = string.atof(al[2])
        venue_lat[venue] = lat
        venue_lng[venue] = lng
    f.close()
    # read checkins
    f = open(checkins_path, 'r')
    u_checkins = {}
    cnt = 0
    # lines = f.readlines()
    for line in f:
        checkin = CheckIn(line.strip(), venue_lat, venue_lng)
        if checkin.isvalid():
            if checkin.user not in u_checkins:
                u_checkins[checkin.user] = []
            u_checkins[checkin.user].append(checkin)
            cnt += 1
            if cnt % 1000 == 0:
                print 'reading: %d' % cnt
    f.close()
    # threads = []
    # threads_num = 8
    # per_thread = len(lines) / threads_num
    # print per_thread
    # for i in range(threads_num):
    #     if threads_num - i == 1:
    #         t = threading.Thread(target=process, args=(i, lines[i * per_thread:], venue_lat, venue_lng, u_checkins))
    #     else:
    #         t = threading.Thread(target=process, args=(i, lines[i * per_thread:i * per_thread + per_thread], venue_lat, venue_lng, u_checkins))
    #     threads.append(t)
    # for i in range(threads_num):
    #     threads[i].start()
    # for i in range(threads_num):
    #     threads[i].join()
    checkins_all = []
    for checkins_u in u_checkins.values():
        checkins_u = sorted(checkins_u, cmp=lambda x, y: cmp(x.dt, y.dt))
        for checkin in checkins_u:
            checkins_all.append(checkin)
    f = open(save_path, 'w')
    for checkin in checkins_all:
        f.write(checkin.to_string() + '\n')
    f.close()

mutex = threading.Lock()
def process(tid, lines, venue_lat, venue_lng, u_checkins):
    for cnt, line in enumerate(lines):
        checkin = CheckIn(line.strip(), venue_lat, venue_lng)
        if checkin.isvalid():
            if mutex.acquire(1):
                if checkin.user not in u_checkins:
                    u_checkins[checkin.user] = []
            u_checkins[checkin.user].append(checkin)
            mutex.release()
        if cnt % 1000 == 0:
            print 'thread: ', tid, 'line: ', cnt

class CheckIn(object):
    def __init__(self, str, venue_lat, venue_lng):
        try:
            # print str
            self.is_valid = True
            al = str.split(",")
            self.user = al[7].strip('"')
            self.venue = al[8].strip('"')
            if self.venue not in venue_lng.keys():
                self.is_valid = False
            self.lat = venue_lat[self.venue]
            self.lng = venue_lng[self.venue]
            self.dt = datetime.datetime.strptime(al[1].strip('"'), '%Y-%m-%d %H:%M:%S')
            self.time = al[1].strip('"').replace(' ', 'T') + 'Z'
        except:
            self.is_valid = False

    def isvalid(self):
        return self.is_valid

    def to_string(self):
        return self.user + '\t' + self.time + '\t' + str(self.lat) + '\t' + str(self.lng) + '\t' + self.venue