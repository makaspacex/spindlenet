import sys

import lmdb
import six
from PIL import Image


class NekoOcrLmdbMgmt:
    def __init__(self, root, data_filtering_off, batch_max_length):
        self.env = lmdb.open(root, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)
        self.data_filtering_off = data_filtering_off
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            if self.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    # neko_lmdb starts with 0
                    label_key = 'label-%09d'.encode() % index
                    try:
                        label = txn.get(label_key).decode('utf-8')
                    except:
                        continue
                    if len(label) > batch_max_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset_related {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    # out_of_char = f'[^{self.opt.character}]'
                    # if re.search(out_of_char, label.lower()):
                    #     continue

                    self.filtered_index_list.append(index)
                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def get_pair(self, txn, img_key, label_key):
        try:
            label = txn.get(label_key).decode('utf-8')
        except:
            print(f'??? for ', label_key)
            return None, None
        if (label == "###"):
            print(f'IGN for ', label_key)
            return None, None

        imgbuf = txn.get(img_key)
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            img = Image.open(buf)
        except IOError:
            print(f'Corrupted image for ', img_key)
            # make dummy image and dummy label for corrupted image.
            return None, None

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            # out_of_char = f'[^{self.opt.character}]'
            # label = re.sub(out_of_char, '', label)
        return (img, label)

    def justify_idx(self, index):
        return self.filtered_index_list[index % len(self)]

    def get_encoded_im_by_name(self, name):
        try:
            with self.env.begin(write=False) as txn:
                imgbuf = txn.get(name)
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf).convert('RGB')  # for color image
        except:
            img = None
        return img

    def getitem_encoded_im(self, index):
        index = self.justify_idx(index)

        assert index <= len(self), 'index range error'
        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            try:
                label = txn.get(label_key).decode('utf-8')
            except:
                return None, None
            if (label == "###"):
                return None, None
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
        return (imgbuf, label)

    def getitem_kv(self, index, imks, tks, rks):
        index = self.justify_idx(index)

        assert index <= len(self), 'index range error'
        with self.env.begin(write=False) as txn:
            imgbufs = []
            for ik in imks:
                img_key = (ik + '-%09d').encode() % index
                try:
                    imgbuf = txn.get(img_key)

                    buf = six.BytesIO()
                    buf.write(imgbuf)
                    buf.seek(0)
                    img = Image.open(buf).convert('RGB')  # for color image
                    imgbufs.append(img)
                except:
                    imgbufs.append(None)

            labels = []
            for lk in tks:
                label_key = (lk + '-%09d').encode() % index
                try:
                    labels.append(txn.get(label_key).decode('utf-8'))
                except:
                    labels.append(None)
            rawbufs = []
            for rk in rks:
                img_key = (rk + '-%09d').encode() % index
                try:
                    rawbufs.append(txn.get(img_key))
                except:
                    rawbufs.append(None)

        return imgbufs, labels, rawbufs

    def getitem_encoded_kv(self, index, imks, tks):
        index = self.justify_idx(index)

        assert index <= len(self), 'index range error'
        with self.env.begin(write=False) as txn:
            labels = []
            for lk in tks:
                label_key = (lk + '-%09d').encode() % index
                try:
                    labels.append(txn.get(label_key).decode('utf-8'))
                except:
                    labels.append(None)
            imgbufs = []
            for ik in imks:
                img_key = (ik + '-%09d').encode() % index
                try:
                    imgbufs.append(txn.get(img_key))
                except:
                    imgbufs.append(None)

        return imgbufs, labels

    def parse_to_dict(self, ks, cs):
        d = {}
        for i in range(len(ks)):
            d[ks[i]] = cs[i]
        return d

    def get_indexed(self, index):
        index = self.justify_idx(index)
        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            img_key = 'image-%09d'.encode() % index
            img, text = self.get_pair(txn, img_key, label_key)
        if (img is None):
            return None, None
        return (img, text)
