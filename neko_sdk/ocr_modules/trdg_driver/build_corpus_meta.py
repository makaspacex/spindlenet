import glob
import os

import torch

from neko_sdk.lmdb_wrappers.corpus_lmdb_wrapper import CorpusLmdbWrapper
from neko_sdk.ocr_modules.fontkit.fntmgmt import Fntmgmt


class NekoMetaBuilderType1:

    def index_fonts(self, font_folder, langs):
        self.lang_dict = {}
        self.fnts = []
        self.fnt_cs = []
        for l in langs:
            self.lang_dict[l] = []
            path = os.path.join(font_folder, l)
            fnts = glob.glob(os.path.join(path, "*.*"))
            for f in fnts:
                self.lang_dict[l].append(len(self.fnts))
                self.fnts.append(f)
                self.fnt_cs.append(Fntmgmt.get_charset(f))

    def scan_content(self, content):
        if (len(content) == 0):
            return
        sc = set(content)
        fcids = []
        for fid in range(len(self.fnts)):
            if sc == sc.intersection(self.fnt_cs[fid]):
                fcids.append(fid)
        if (len(fcids)):
            self.db.add_data_utf(content, fcids)
            self.valid_cnt += 1
        else:
            print("Nothing renders $C$: ", content[:28])
            self.invalid_cnt += 1
            print(self.valid_cnt, " vs ", self.invalid_cnt)

    def scan_corpus(self, corpus_folder):
        self.valid_cnt = 0
        self.invalid_cnt = 0

        self.content = []
        self.fnt_content = [[] for _ in self.fnts]
        files = glob.glob(os.path.join(corpus_folder, "*.txt"))
        for f in files:
            with open(f, "r") as fp:
                for l in fp:
                    self.scan_content(l.strip())
                    # if(self.valid_cnt>100):
                    #     break
        self.db.end_this()

    def get_meta(self, corpus_folder, font_folder, langs):
        self.db = CorpusLmdbWrapper("corpusdb")
        self.langs = []
        self.index_fonts(font_folder, langs)
        self.scan_corpus(corpus_folder)
        meta = {}
        meta["fonts"] = self.fnts
        torch.save(meta, "fonts.pt")


engine = NekoMetaBuilderType1()
engine.get_meta("/home/lasercat/netdata/corpus/", "/home/lasercat/ssddata/mltnocr/fonts/", ["ch", "jp"])
