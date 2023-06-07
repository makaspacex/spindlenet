import glob
import os.path
import shutil

import cv2

from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import ImLmdbWrapper

def quick_lmdb(srcpath,dst,lang="None",imgpfix="png"):
    samples_gt=glob.glob(os.path.join(srcpath,"*.txt"));
    shutil.rmtree(dst,True);
    db=ImLmdbWrapper(dst);
    for gname in samples_gt:
        iname=gname.replace("txt",imgpfix);
        img=cv2.imread(iname);
        if(img is None):
            print("bad", iname,gname);
            continue;
        with open(gname,"r") as fp:
            anno=[_.strip() for _ in fp][0];
        db.add_data_utf(img,anno,lang);
    db.end_this();
    print("debug");