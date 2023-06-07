import pickle
import random
from builtins import list

import lmdb

from neko_sdk.thirdparty.trdg.data_generator import FakeTextDataGenerator


class NekoAbstractStringGenerator:
    def mount_meta(self, meta):
        self.fnt_charset = meta["fnt_charset"]
        self.fnt_grp = meta["fnt_grp"]
        self.fnt_grp_keys = list(self.fnt_grp.keys())
        self.segments = meta["grpseg"]
        self.segment_length = meta["segment_length"]
        self.spaces = meta["spaces"]
        pass

    def set_genpara(self):
        # generator
        self.gen = FakeTextDataGenerator()
        # hyper parameter
        self.size = 64
        self.skewing_angle = [0, 0, 0, 2, 2, 5, 5, 10]
        self.random_skew = [False, False, False, True]
        self.blur = [0, 1]
        self.random_blur = [True, True, False]
        self.background_types = [0, 1, 2, 3, 3, 3, 3]
        self.distorsion_type = [False, False, 0, 1, 2]
        self.distorsion_orientation = [0, 1, 2]
        self.is_handwritten = False
        self.width = -1
        self.alignment = 0
        self.text_color = "#010101"
        self.orientation = 0
        self.space_width = 1
        self.character_spacing = [0, 64, 32]
        self.margins = (5, 5, 5, 5)
        self.fit = 0
        self.output_mask = 0
        self.word_split = [True, False]

    def __init__(self, meta, bgims):
        # meta info
        self.mount_meta(meta)
        self.bgims = bgims
        # background image list
        self.set_genpara()
        pass

    def get_content(self):
        return None, None

    # Drive the vehicle
    def drive(self, bgtype, bgim, font, content):
        size = self.size
        skewing_angle = random.choice(self.skewing_angle)
        random_skew = random.choice(self.random_skew)
        blur = random.choice(self.blur)
        random_blur = self.random_blur
        background_type = bgtype
        distorsion_type = random.choice(self.distorsion_type)
        distorsion_orientation = random.choice(self.distorsion_orientation)
        is_handwritten = self.is_handwritten
        width = self.width
        alignment = self.alignment
        if (bgim is not None):
            text_color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        else:
            text_color = "#" + ''.join([random.choice('0123456789') for j in range(6)])

        orientation = self.orientation
        space_width = self.space_width
        character_spacing = random.choice(self.character_spacing)
        margins = self.margins
        fit = self.fit
        if (len(content) <= 5):
            word_split = random.choice(self.word_split)
        else:
            word_split = True
        final_image, final_mask = self.gen.generate_core(
            content,
            font,
            size,
            skewing_angle,
            random_skew,
            blur,
            random_blur,
            background_type,
            distorsion_type,
            distorsion_orientation,
            is_handwritten,
            width,
            alignment,
            text_color,
            orientation,
            space_width,
            character_spacing,
            margins,
            fit,
            word_split,
            bgim,
        )
        return final_image, content, final_mask

    def random_bgm(self):
        btype = random.choice(self.background_types)
        bgim = None
        if (btype == 3):
            try:
                bgim = random.choice(self.bgims)
            except:
                return self.random_bgm()
            if (bgim is None):
                return self.random_bgm()
        return btype, bgim

    def random_clip(self):
        fnt, content = self.get_content()
        bgtype, bgm = self.random_bgm()
        final_image, content, final_mask = self.drive(bgtype, bgm, fnt, content)
        return final_image, content


class NekoRandomStringGenerator(NekoAbstractStringGenerator):
    def __init__(self, meta, bgims, max_len):
        super(NekoRandomStringGenerator, self).__init__(meta, bgims, max_len)
        self.max_len = max_len
        pass

    def mount_meta(self, meta):
        self.fnt_charset = meta["fnt_charset"]
        self.fnt_grp = meta["fnt_grp"]
        self.fnt_grp_keys = list(self.fnt_grp.keys())
        self.segments = meta["grpseg"]
        self.segment_length = meta["segment_length"]
        self.spaces = meta["spaces"]

    def random_str(self, charset, l):
        return ''.join(random.choice(charset) for _ in range(l))

    def compose_content(self, segment_lens, charset, space):
        str = "".join(self.random_str(charset, l) + space for l in segment_lens)
        return str.strip(space)[:self.max_len]

    def get_content(self):
        fntg = random.choice(self.fnt_grp_keys)
        fnt = random.choice(self.fnt_grp[fntg])
        charset = self.fnt_charset[fnt]
        segment_cnt = random.choice(self.segments[fntg])
        segments_length = [random.choice(self.segment_length[fntg]) for _ in range(segment_cnt)]
        content = self.compose_content(segments_length, charset, space=self.spaces[fntg])
        if (len(content) == 0):
            return self.get_content()
        return fnt, content


class NekoSkipMissingStringGenerator(NekoAbstractStringGenerator):
    def __init__(self, meta, bgims, max_len):
        super(NekoSkipMissingStringGenerator, self).__init__(meta, bgims, max_len)
        self.max_len = max_len
        pass

    def mount_meta(self, meta):
        self.fnt_charset = meta["fnt_charset"]
        self.fnt_grp = meta["fnt_grp"]
        self.fnt_grp_keys = list(self.fnt_grp.keys())
        corpusdb = meta["corpus_db"]
        self.env = lmdb.open(corpusdb, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        self.nSamples = int(self.txn.get('num-samples'.encode()))
        self.cntr = random.randint(0, self.nSamples)
        # self.segments=meta["grpseg"]
        # self.segment_length=meta["segment_length"]
        # self.spaces=meta["spaces"]

    def compose_content(self, length, charset):
        compatKey = 'content-%09d'.encode() % self.cntr

        rawstr = self.txn.get(compatKey).decode()
        ret = "".join(c if c in charset else "" for c in rawstr)
        return ret[:length]

    def get_content(self):
        self.cntr += 1
        self.cntr %= self.nSamples
        fntg = random.choice(self.fnt_grp_keys)
        fnt = random.choice(self.fnt_grp[fntg])
        charset = self.fnt_charset[fnt]
        # segment_cnt = random.choice(self.segments[fntg])
        # segments_length = [random.choice(self.segment_length[fntg]) for _ in range(segment_cnt)]
        # segments_length =random.choice(self.segment_length[fntg])
        content = self.compose_content(self.max_len, charset)
        if (len(content) == 0):
            return self.get_content()
        return fnt, content


class NekoRandomCorpusGenerator(NekoAbstractStringGenerator):
    def mount_meta(self, meta, corpusdb):
        self.fonts = meta["fonts"]
        self.env = lmdb.open(corpusdb, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        self.nSamples = int(self.txn.get('num-samples'.encode()))
        self.cntr = 0

    def __init__(self, meta, corpusdb, bgims):
        # meta info
        self.mount_meta(meta, corpusdb)
        self.bgims = bgims
        # background image list
        self.set_genpara()
        pass

    def get_content_idx(self, idx):
        contentKey = 'content-%09d'.encode() % idx
        compatKey = 'compatible-%09d'.encode() % idx
        content = self.txn.get(contentKey).decode()
        compat_list = pickle.loads(self.txn.get(compatKey))
        fnt = self.fonts[random.choice(compat_list)]
        return fnt, content

    def _get_content(self):
        self.cntr += 1
        self.cntr %= self.nSamples
        fnt, content = self.get_content_idx(self.cntr)
        # try:
        #
        # except:
        #     return self.get_content()
        return fnt, content[:25]

    def get_content(self):
        try:
            return self._get_content()
        except:
            return self._get_content()
