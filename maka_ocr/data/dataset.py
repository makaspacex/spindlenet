from torch.utils.data import Dataset
import random
import sys
import torchvision.transforms as transforms
import cv2
import lmdb
import numpy as np
import six
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from maka_ocr.data.processes.data_process import DataProcess

class OCRLmdbDataset(Dataset):
    def __init__(
        self, db_root: str, meta_path: str, transform: list = None, maxT = 40, data_process:DataProcess=None
    ) -> None:
        super().__init__()
        self.db_root = db_root
        self.meta_path = meta_path
        self.transform = transform
        self.maxT = maxT
        self.set_dss(db_root)
        self.data_process = data_process
        
    def set_dss(self, db_root: list[DataProcess]):
        env = lmdb.open(
            db_root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not env:
            raise Exception("cannot creat lmdb from %s" % (db_root))
        txn = env.begin(write=False)
        keys = list(txn.cursor().iternext(values=False))
        kk = [
            int(str(_.decode()).split("-")[-1])
            for _ in keys
            if "num" not in str(_.decode())
        ]
        index_num_list = sorted(list(set(kk)))
        index_num_list = [int(k) for k in index_num_list]

        self.txn = txn
        self.env = env
        self.index_num_list = index_num_list

    def __len__(self):
        return len(self.index_num_list)

    def __getitem__(self, item):
        index = self.index_num_list[item]
        txn = self.txn
        img_key = "image-%09d" % index
        try:
            imgbuf = txn.get(img_key.encode())
            img = cv2.imdecode(np.frombuffer(imgbuf, np.uint8),  cv2.IMREAD_COLOR)
            # img = cv2.imdecode(np.array(bytearray(imgbuf), dtype='uint8'), cv2.IMREAD_COLOR)
        except:
            print("Corrupted image for %d" % index)
            return self[index + 1]

        label_key = "label-%09d" % index
        label = str(txn.get(label_key.encode()).decode("utf-8"))

        if len(label) > self.maxT - 1:
            print(f"sample too long: {label}")
            return self[index + 1]

        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        
        data = {"image": img, "label": label}
        
        # 处理数据
        if self.data_process:
            data = self.data_process(data)
        
        # 处理图像的transform
        if self.transform:
            data['image'] = self.transform(data['image'])
            if "bmask" in data:
                data['bmask'] = self.transform(data['bmask'])
        
        return data


if __name__ == "__main__":
    from maka_ocr.data.processes.keep_ratio_resize_for_mth import MakeImgCenterAndPad
    from maka_ocr.data import processes
    
    img_hw = [320,32]
    process = processes.Compose([MakeImgCenterAndPad(img_hw=img_hw)])
    tkh_train_dataset = OCRLmdbDataset(db_root="recdatassd/TKH_train", meta_path="ddd", data_process=process)
    
    for ss in tkh_train_dataset:
        print(ss)
