import os

from dataloaders import NekoGfslEvalTaskDataset
from dataloaders import NekoFslNoProtoTaskDatasetg2
from dataloaders import RandomSampler
from neko_sdk.ds_meta.oracle_1 import all


def get_orahw_tr_meta(root):
    return os.path.join(root, "dicts", "dab_oracle_tr.pt")


def get_orahw_te_meta(root):
    return os.path.join(root, "dicts", "dab_oracle_te.pt")


def get_orahw_train(root):
    trroot = os.path.join(root, "oraclehw")
    rep = 1
    return \
        {
            "type": NekoFslNoProtoTaskDatasetg2,
            'ds_args':
                {
                    "db_root": trroot,
                    "waycnt": 64,
                    "ucnt": 16,
                    "labelset": all[:3400],
                    "dsize": [32, 32],
                },
            'dl_args':
                {
                    'batch_size': 1,
                    "sampler": RandomSampler(None),
                    'num_workers': 5,
                },
        }


def get_eval_orahw_core(root):
    teroot = os.path.join(root, "oraclehw")
    return \
        {
            "type": NekoGfslEvalTaskDataset,
            'ds_args':
                {
                    'db_root': teroot,
                    'labelset': all[3400:],
                    'dsize': [32, 32],
                },
            'dl_args':
                {
                    'batch_size': 64,
                    'shuffle': False,
                    'num_workers': 5,
                },
        }


def get_eval_orahw(root):
    return {
        "dict_dir": None,
        "case_sensitive": False,
        "te_case_sensitive": False,
        "datasets": {
            "jrc_traffic_eval": get_eval_orahw_core(root),
        }
    }
