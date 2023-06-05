from neko_2021_mjt.configs.data.mth_data import *
from neko_2020nocr.dan.configs.datasets import ds_paths 
from dataloaders import NekoJointLoader


# ----------------- eval_dss -------------------------
def get_mth1000_eval_dss(dsroot, maxT, batchsize=32, hw=[320, 32]):
    te_meta_path = get_mth1000_test_all_meta(dsroot)
    eval_ds = get_test_mth1000_dsrgb(
        maxT=maxT, dsroot=dsroot, hw=hw, batchsize=batchsize
    )
    return te_meta_path, eval_ds


def get_mth1200_eval_dss(dsroot, maxT, batchsize=32, hw=[320, 32]):
    te_meta_path = get_mth1200_test_all_meta(dsroot)
    eval_ds = get_test_mth1200_dsrgb(
        maxT=maxT, dsroot=dsroot, hw=hw, batchsize=batchsize
    )
    return te_meta_path, eval_ds

def get_tkh_eval_dss(dsroot, maxT, batchsize=32, hw=[320, 32]):
    te_meta_path = get_tkh_test_all_meta(dsroot)
    eval_ds = get_test_tkh_dsrgb(maxT=maxT, dsroot=dsroot, hw=hw, batchsize=batchsize)
    return te_meta_path, eval_ds


def get_tkhmth2200_eval_dss(dsroot, maxT, batchsize=32, hw=[320, 32]):
    te_meta_path = get_tkhmth2200_test_all_meta(dsroot)
    eval_ds = get_test_tkhmth2200_dsrgb(
        maxT=maxT, dsroot=dsroot, hw=hw, batchsize=batchsize
    )
    return te_meta_path, eval_ds

# ----------------- train_dss -------------------------
def get_dataloadercfgs(train_name, train_cfg):
    return {
        "loadertype": NekoJointLoader,
        "subsets": {
            # "dan_mth1200": get_mth1200_train_cfg(root, maxT, bs=bsize, hw=hw),
            train_name: train_cfg,
        },
    }

def get_mth1000_train_dss(dsroot, maxT, bsize, hw=[320, 32]):
    te_meta_path, eval_ds = get_mth1000_eval_dss(dsroot, maxT)
    tr_meta_path = get_mth1000_train_meta(dsroot)
    train_name = "dan_mth1000"
    dsroots = [ds_paths.get_mth1000_train(dsroot)]
    dataloader_cfgs = get_train_dataloader_cfgs(train_name=train_name, dsroots=dsroots, maxT=maxT, bsize=bsize, hw=hw, random_aug=False )
    return tr_meta_path, dataloader_cfgs, eval_ds, te_meta_path

def get_mth1200_train_dss(dsroot, maxT, bsize, hw=[320, 32]):
    te_meta_path, eval_ds = get_mth1200_eval_dss(dsroot, maxT)
    tr_meta_path = get_mth1200_train_meta(dsroot)
    train_name = "dan_mth1200"
    dsroots = [ds_paths.get_mth1200_train(dsroot)]
    dataloader_cfgs = get_train_dataloader_cfgs(train_name=train_name, dsroots=dsroots, maxT=maxT, bsize=bsize, hw=hw, random_aug=False )
    return tr_meta_path, dataloader_cfgs, eval_ds, te_meta_path


def get_tkh_train_dss(dsroot, maxT, bsize, hw=[320, 32]):
    te_meta_path, eval_ds = get_tkh_eval_dss(dsroot, maxT)
    tr_meta_path = get_tkh_train_meta(dsroot)
    train_name = "dan_tkh"
    dsroots = [ds_paths.get_tkh_train(dsroot)]
    dataloader_cfgs = get_train_dataloader_cfgs(train_name=train_name, dsroots=dsroots, maxT=maxT, bsize=bsize, hw=hw, random_aug=False )
    return tr_meta_path, dataloader_cfgs, eval_ds, te_meta_path

def get_tkhmth2200_train_dss(dsroot, maxT, bsize, hw=[320, 32]):
    te_meta_path, eval_ds = get_tkhmth2200_eval_dss(dsroot, maxT)
    tr_meta_path = get_tkhmth2200_train_meta(dsroot)
    train_name = "dan_tkhmth2200"
    dsroots = [ds_paths.get_tkhmth2200_train(dsroot)]
    dataloader_cfgs = get_train_dataloader_cfgs(train_name=train_name, dsroots=dsroots, maxT=maxT, bsize=bsize, hw=hw, random_aug=False )
    return tr_meta_path, dataloader_cfgs, eval_ds, te_meta_path
