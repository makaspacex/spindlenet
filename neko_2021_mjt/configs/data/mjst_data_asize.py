import torch
from torchvision import transforms

from neko_2020nocr.dan.configs.datasets.ds_paths import get_nips14
from neko_2020nocr.dan.configs.datasets.ds_paths import get_cvpr16
from neko_2020nocr.dan.configs.datasets.ds_paths import get_cute
from neko_2020nocr.dan.configs.datasets.ds_paths import get_iiit5k
from neko_2020nocr.dan.configs.datasets.ds_paths import get_SVT
from neko_2020nocr.dan.configs.datasets.ds_paths import get_SVTP
from neko_2020nocr.dan.configs.datasets.ds_paths import get_IC13_1015
from neko_2020nocr.dan.configs.datasets.ds_paths import get_IC03_867
from dataloaders.neko_asize_dataloader import ColoredAsizeLmdbDataset


def asize_collate(batch):
    ret = {}
    # print("a")
    ret["rectimage"] = torch.stack([batch[i]["rectimage"] for i in range(len(batch))])
    ret["label"] = [batch[i]["label"] for i in range(len(batch))]
    ret["rawimage"] = [batch[i]["rawimage"] for i in range(len(batch))]
    return ret


def get_mjstcqaAS_cfg(root, maxT):
    rdic = {
        "type": ColoredAsizeLmdbDataset,
        'ds_args': {
            'roots': [get_nips14(root), get_cvpr16(root)],
            'img_height': 32,
            'img_width': 32,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT": maxT,
            'qhb_aug': True
        },
        "dl_args":
            {
                'batch_size': 48,
                'shuffle': False,
                'num_workers': 3,
                "collate_fn": asize_collate,
            }
    }
    return rdic


def get_datasetAS_testC(maxT, root, dict_dir, batch_size=128):
    return {
        'type': ColoredAsizeLmdbDataset,
        'ds_args': {
            'roots': [root],
            'img_height': 32,
            'img_width': 32,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT,
        },
        'dl_args': {
            "collate_fn": asize_collate,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 4,
        },
    }


def get_test_all_uncased_dsrgbAS(maxT=25, root="/home/lasercat/ssddata/", dict_dir='../../dict/dic_36.txt',
                                 batchsize=128):
    return {
        "dict_dir": dict_dir,
        "case_sensitive": False,
        "te_case_sensitive": False,
        "datasets": {
            "SVTP": get_datasetAS_testC(maxT, get_SVTP(root), dict_dir, batchsize),
            "CUTE": get_datasetAS_testC(maxT, get_cute(root), dict_dir, batchsize),
            "IIIT5k": get_datasetAS_testC(maxT, get_iiit5k(root), dict_dir, batchsize),
            "SVT": get_datasetAS_testC(maxT, get_SVT(root), dict_dir, batchsize),
            "IC03": get_datasetAS_testC(maxT, get_IC03_867(root), dict_dir, batchsize),
            "IC13": get_datasetAS_testC(maxT, get_IC13_1015(root), dict_dir, batchsize),
        }
    }


def get_uncased_dsrgb_d_trAS(maxT=25, root="/home/lasercat/ssddata/", dict_dir='../../dict/dic_36.txt', batchsize=128):
    return {
        "dict_dir": dict_dir,
        "case_sensitive": False,
        "te_case_sensitive": False,
        "datasets": {
            "SVTP": get_datasetAS_testC(maxT, get_nips14(root), dict_dir, batchsize),
        }
    }
