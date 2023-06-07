import os

import cv2
import lmdb


class LmdbWrapper:
    def set_meta(self):
        self.labels = {}
        self.writers = set()

    def __init__(self, lmdb_dir):
        self.root = lmdb_dir
        os.makedirs(lmdb_dir, exist_ok=True)
        self.db = lmdb.open(lmdb_dir, map_size=1e11)
        self.load = 0

    def adddata_kv(self, ikvdict, tkvdict, rkvdict):
        iks = []
        rks = []
        tks = []
        self.txn = self.db.begin(write=True)
        for ik in ikvdict:
            imageKey = (ik + '-%09d' % self.load).encode()
            iks.append(imageKey)
            self.txn.put(imageKey, cv2.imencode(".png", ikvdict[ik])[1])
        for rk in rkvdict:
            rawKey = (rk + '-%09d' % self.load).encode()
            rks.append(rawKey)
            self.txn.put(rawKey, rkvdict[rk])
        for tk in tkvdict:
            tKey = (tk + '-%09d' % self.load).encode()
            tks.append(tKey)
            self.txn.put(tKey, tkvdict[tk].encode())
        self.load += 1
        self.txn.replace('num-samples'.encode(), str(self.load).encode())
        self.txn.commit()
        return iks, rks, tks

    def end_this(self):
        try:
            self.txn = self.db.begin(write=True)
            self.txn.replace('num-samples'.encode(), str(self.load).encode())
            self.txn.commit()
            self.db.close()
        except Exception as e:
            raise e