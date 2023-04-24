from torchvision import transforms

from neko_2020nocr.dan.configs.datasets.ds_paths import *
from neko_2020nocr.dan.dataloaders.dataset_scene import ColoredLmdbDataset, ColoredLmdbDatasetT


def get_quickviet_training_cfg(root, maxT, bs=48, hw=[32, 128], random_aug=True):
    rdic = {
        "type": ColoredLmdbDataset,
        'ds_args': {
            'roots': ["/run/media/lasercat/writebuffer/quickviet/lmdb/"],
            'img_height': hw[0],
            'img_width': hw[1],
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT": maxT,
            'qhb_aug': random_aug
        },
        "dl_args":
            {
                'batch_size': bs,
                'shuffle': False,
                'num_workers': 8,
            }
    }
    return rdic


def get_quickviet_testC(maxT, root, dict_dir, batch_size=128, hw=[32, 128]):
    return {
        'type': ColoredLmdbDatasetT,
        'ds_args': {
            'roots': ["/run/media/lasercat/writebuffer/quickviet/lmdb/"],
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


def get_test_quickviet_uncased_dsrgb(maxT=25, root="/home/lasercat/ssddata/", dict_dir=None, batchsize=128,
                                     hw=[32, 128]):
    return {
        "dict_dir": dict_dir,
        "case_sensitive": False,
        "te_case_sensitive": False,
        "datasets": {
            "quickviet": get_quickviet_testC(maxT, get_cute(root), dict_dir, batchsize, hw),
        }
    }
