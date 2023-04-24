import pickle

from neko_sdk.lmdb_wrappers.lmdb_wrapper import lmdb_wrapper


class corpus_lmdb_wrapper(lmdb_wrapper):
    def __init__(self, dbpath):
        super(corpus_lmdb_wrapper, self).__init__(dbpath)

    def add_data_utf(self, content, compatible_list):
        contentKey = 'content-%09d'.encode() % self.load

        self.txn.put(contentKey, content.encode())
        if (compatible_list is not None):
            compatKey = 'compatible-%09d'.encode() % self.load
            self.txn.put(compatKey, pickle.dumps(compatible_list))
        if (self.load % 500 == 0):
            self.txn.replace('num-samples'.encode(), str(self.load).encode())
            print("load:", self.load)
            self.txn.commit()
            del self.txn
            self.txn = self.db.begin(write=True)
        self.load += 1
