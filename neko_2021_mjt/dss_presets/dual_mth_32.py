from neko_2021_mjt.configs.data.mth_data import *
from dataloaders import NekoJointLoader

def get_dataloadercfgs(root, maxT,bsize, hw=[320,32]):
    return {
            "loadertype": NekoJointLoader,
            "subsets":
                {
                    "dan_mth1200": get_mth1200_train_cfg(root, maxT, bs=bsize, hw=hw),
                }
        }

# ----------------- eval_dss -------------------------
def get_mth1000_eval_dss(dsroot, maxT, batchsize=32, hw=[320,32]):
    te_meta_path = get_mth1000_test_all_meta(dsroot);
    eval_ds = get_test_mth1000_dsrgb(maxT=maxT,dsroot=dsroot, hw=hw, batchsize=batchsize);
    return te_meta_path, eval_ds
def get_mth1200_eval_dss(dsroot, maxT, batchsize=32, hw=[320,32]):
    te_meta_path = get_mth1200_test_all_meta(dsroot);
    eval_ds = get_test_mth1200_dsrgb(maxT=maxT,dsroot=dsroot, hw=hw, batchsize=batchsize);
    return te_meta_path, eval_ds
def get_tkh_eval_dss(dsroot, maxT, batchsize=32, hw=[320,32]):
    te_meta_path = get_tkh_test_all_meta(dsroot);
    eval_ds = get_test_tkh_dsrgb(maxT=maxT,dsroot=dsroot, hw=hw, batchsize=batchsize);
    return te_meta_path, eval_ds
def get_tkhmth2200_eval_dss(dsroot, maxT, batchsize=32, hw=[320,32]):
    te_meta_path = get_tkhmth2200_test_all_meta(dsroot);
    eval_ds = get_test_tkhmth2200_dsrgb(maxT=maxT,dsroot=dsroot, hw=hw, batchsize=batchsize);
    return te_meta_path, eval_ds

# ----------------- train_dss -------------------------
def get_mth1000_train_dss(dsroot, maxT, bsize, hw=[320,32]):
    te_meta_path, eval_ds = get_mth1000_eval_dss(dsroot, maxT)
    tr_meta_path = get_mth1000_train_meta(dsroot)
    train_joint_ds = get_dataloadercfgs(dsroot=dsroot, maxT=maxT, bsize=bsize, hw=hw)
    return tr_meta_path, train_joint_ds,eval_ds,te_meta_path
def get_mth1200_train_dss(dsroot, maxT, bsize, hw=[320,32]):
    te_meta_path, eval_ds = get_mth1200_eval_dss(dsroot, maxT)
    tr_meta_path = get_mth1200_train_meta(dsroot)
    train_joint_ds = get_dataloadercfgs(dsroot=dsroot, maxT=maxT, bsize=bsize, hw=hw)
    return tr_meta_path, train_joint_ds,eval_ds,te_meta_path
def get_tkh_train_dss(dsroot, maxT, bsize, hw=[320,32]):
    te_meta_path, eval_ds = get_tkh_eval_dss(dsroot, maxT)
    tr_meta_path = get_tkh_train_meta(dsroot)
    train_joint_ds = get_dataloadercfgs(dsroot=dsroot, maxT=maxT, bsize=bsize, hw=hw)
    return tr_meta_path, train_joint_ds,eval_ds,te_meta_path
def get_tkhmth2200_train_dss(dsroot, maxT, bsize, hw=[320,32]):
    te_meta_path, eval_ds = get_tkhmth2200_eval_dss(dsroot, maxT)
    tr_meta_path = get_tkhmth2200_train_meta(dsroot)
    train_joint_ds = get_dataloadercfgs(dsroot=dsroot, maxT=maxT, bsize=bsize, hw=hw)
    return tr_meta_path, train_joint_ds,eval_ds,te_meta_path
