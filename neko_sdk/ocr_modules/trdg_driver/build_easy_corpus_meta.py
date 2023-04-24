import glob
import os

from neko_sdk.lmdb_wrappers.corpus_lmdb_wrapper import corpus_lmdb_wrapper
from neko_sdk.ocr_modules.fontkit.fntmgmt import fntmgmt


class neko_meta_builder_type1:

    def scan_content(self, content):
        self.db.add_data_utf(content, None)
        self.valid_cnt += 1

    def scan_corpus(self, corpus_folder):
        self.valid_cnt = 0

        self.content = []
        files = glob.glob(os.path.join(corpus_folder, "*.txt"))
        for f in files:
            with open(f, "r") as fp:
                for l in fp:
                    if (len(l.strip().replace(" ", "")) == 0):
                        continue
                    if (set(l.strip()).intersection(self.all_valid) != set(l.strip())):
                        # print(l)
                        continue

                    self.scan_content(l.strip())
                    # if(self.valid_cnt>100):
                    #     break
        self.db.end_this()

    def setup(self, corpus_folder, dbpath):
        self.all_valid = fntmgmt.get_charset('../../stdfnts/NotoSansCJK-Regular.ttc')
        self.db = corpus_lmdb_wrapper(dbpath)
        self.scan_corpus(corpus_folder)


neko_meta_builder_type1().setup("/home/lasercat/corpus", "/home/lasercat/corpus/corpusdb")
