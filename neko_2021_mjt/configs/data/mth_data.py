from torchvision import transforms
import os
from neko_2020nocr.dan.configs.datasets import ds_paths 
from neko_2020nocr.dan.dataloaders.dataset_scene import ColoredLmdbDatasetV, ColoredLmdbDatasetTV
from dataloaders import NekoJointLoader

# ------------------ meta ---------------------------------------
def get_mth1000_test_all_meta(root):
    temeta = os.path.join(root, "MTH1000_test_all", "dict_test_all.pt")
    return temeta

def get_mth1000_train_meta(root):
    temeta = os.path.join(root, "MTH1000_train", "dict_train.pt")
    return temeta

def get_mth1200_test_all_meta(root):
    temeta = os.path.join(root, "MTH1200_test_all", "dict_test_all.pt")
    return temeta

def get_mth1200_train_meta(root):
    temeta = os.path.join(root, "MTH1200_train", "dict_train.pt")
    return temeta

def get_tkh_test_all_meta(root):
    temeta = os.path.join(root, "TKH_test_all", "dict_test_all.pt")
    return temeta

def get_tkh_train_meta(root):
    temeta = os.path.join(root, "TKH_train", "dict_train.pt")
    return temeta

def get_tkhmth2200_test_all_meta(root):
    temeta = os.path.join(root, "TKHMTH2200_test_all", "dict_test_all.pt")
    return temeta

def get_tkhmth2200_train_meta(root):
    temeta = os.path.join(root, "TKHMTH2200_train", "dict_train.pt")
    return temeta

# ------------------ train_cfg ---------------------------------------
def _get_base_train_cfg(dsroots:list, maxT, bsize=48, hw=(128, 32), qhb_aug=False):
    rdic = {
        "type": ColoredLmdbDatasetV,
        'ds_args': {
            "roots": dsroots,
            'img_height': hw[0],
            'img_width': hw[1],
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT": maxT,
            'qhb_aug': qhb_aug
        },
        "dl_args":
            {
                'batch_size': bsize,
                'shuffle': True,
                'num_workers': 8,
            }
    }
    return rdic

def get_train_dataloader_cfgs(train_name, dsroots, maxT, bsize=48, hw=(128, 32), qhb_aug=False):
    train_cfg = _get_base_train_cfg(dsroots, maxT, bsize=bsize, hw=hw, qhb_aug=qhb_aug)
    return {
        "loadertype": NekoJointLoader,
        "subsets": {
            train_name: train_cfg,
        },
    }

# ------------------ test_cfg ---------------------------------------
def get_dataset_testC(maxT, root, dict_dir, batch_size=128, hw=(32, 128)):
    return {
        'type': ColoredLmdbDatasetTV,
        'ds_args': {
            'roots': [root],
            'img_height': hw[0],
            'img_width': hw[1],
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT,
        },
        'dl_args': {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 8,
        },
    }

def _get_base_test_cfg(maxT=25, dict_dir=None, batchsize=128, hw=(128, 32), ds_name="", ds_root=None):
    assert ds_root != None, "ds_root required"
    return {
        "dict_dir": dict_dir,
        "case_sensitive": False,
        "te_case_sensitive": False,
        "datasets": {
            ds_name: get_dataset_testC(maxT, ds_root, dict_dir, batchsize, hw),
        }
    }

def get_test_mth1000_dsrgb(maxT=25, dsroot=None, dict_dir=None, batchsize=128, hw=(128, 32)):
    ds_root = ds_paths.get_mth1000_test_all(dsroot)
    ds_name = "MTH1000"
    res_dict = _get_base_test_cfg(maxT=maxT, dict_dir=dict_dir, batchsize=batchsize, hw=hw, ds_name=ds_name, ds_root=ds_root)
    return res_dict

def get_test_mth1200_dsrgb(maxT=25, dsroot=None, dict_dir=None, batchsize=128, hw=(128, 32)):
    ds_root = ds_paths.get_mth1200_test_all(dsroot)
    ds_name = "MTH1200"
    res_dict = _get_base_test_cfg(maxT=maxT, dict_dir=dict_dir, batchsize=batchsize, hw=hw, ds_name=ds_name, ds_root=ds_root)
    return res_dict

def get_test_tkh_dsrgb(maxT=25, dsroot=None, dict_dir=None, batchsize=128, hw=(128, 32)):
    ds_root = ds_paths.get_tkh_test_all(dsroot)
    ds_name = "TKH1200"
    res_dict = _get_base_test_cfg(maxT=maxT, dict_dir=dict_dir, batchsize=batchsize, hw=hw, ds_name=ds_name, ds_root=ds_root)
    return res_dict

def get_test_tkhmth2200_dsrgb(maxT=25, dsroot=None, dict_dir=None, batchsize=128, hw=(128, 32)):
    ds_root = ds_paths.get_tkhmth2200_test_all(dsroot)
    ds_name = "TKHMTH1200"
    res_dict = _get_base_test_cfg(maxT=maxT, dict_dir=dict_dir, batchsize=batchsize, hw=hw, ds_name=ds_name, ds_root=ds_root)
    return res_dict