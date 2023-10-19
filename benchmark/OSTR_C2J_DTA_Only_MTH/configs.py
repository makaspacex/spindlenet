from neko_2021_mjt.configs.loadouts.base_module_set import arm_base_routine2
from neko_2021_mjt.configs.loadouts.base_module_set import arm_base_task_default2
from neko_2021_mjt.configs.loadouts.mk7.lsctt_module_set_mk7 import arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7
from neko_2021_mjt.configs.loadouts.mk7.lsctt_module_set_mk7 import arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7_wide_64
from neko_2021_mjt.configs.loadouts.mk7.lsctt_module_set_mk7 import arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7_wide_64_layernorm
from neko_2021_mjt.configs.routines.ocr_routines.mk7.osdanmk7_routine_cfg import osdanmk7_eval_routine_cfg
from neko_2021_mjt.configs.routines.ocr_routines.mk7.osdanmk7_routine_cfg import osdanmk7dt_ocr_routine
from neko_2021_mjt.configs.routines.ocr_routines.mk7.osdanmk7_routine_cfg import osdanmk7_ocr_routine
from neko_2021_mjt.dss_presets.dual_mth_32 import *

def model_mod_cfg(tr_meta_path, maxT, feat_ch = 768):
    capacity = 512
    mods = {}
    prefix="base_mth1200_"
    # mods = arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7(
    #     prefix=prefix,
    #     maxT=maxT,
    #     capacity=capacity,
    #     feat_ch=feat_ch,
    #     tr_meta_path=tr_meta_path,
    #     srcdst=None,
    #     wemb=0,
    # )
    
    mods = arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7_wide_64(
        prefix=prefix,
        maxT=maxT,
        capacity=capacity,
        feat_ch=feat_ch,
        tr_meta_path=tr_meta_path,
        srcdst=None,
        wemb=0,
    )
    return mods


def _get_base_dan_single_model_train_cfg(prefix, ds_prefix, train_dss, save_root, log_path, log_each, maxT = 40, itrk="Top Nep", bsize=48, tvitr=200000, capacity = 512, feat_ch = 768 ):
    # tr_meta_path, dataloader_cfgs, eval_ds, te_meta_path = get_mth1200_train_dss(dsroot, maxT, bsize)
    tr_meta_path, dataloader_cfgs, eval_ds, te_meta_path = train_dss
    task_dict = {}
    task_dict = arm_base_task_default2(task_dict, prefix, osdanmk7_eval_routine_cfg, 
                                       maxT, 
                                       te_meta_path,
                                       eval_ds,
                                       log_path)

    routines = {}
    routines = arm_base_routine2(routines, prefix, osdanmk7_ocr_routine, maxT, log_path,log_each, ds_prefix)
    # modules = arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7(prefix=prefix, maxT=maxT, capacity=capacity, feat_ch=feat_ch, tr_meta_path=tr_meta_path, srcdst=None, wemb=0)
    
    # modules = arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7_wide_64(prefix=prefix, maxT=maxT, capacity=capacity, feat_ch=feat_ch, tr_meta_path=tr_meta_path, srcdst=None, wemb=0)
    modules = arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7_wide_64_layernorm(prefix=prefix, maxT=maxT, capacity=capacity, feat_ch=feat_ch, tr_meta_path=tr_meta_path, srcdst=None, wemb=0)
    
    
    return {
            "root": save_root,
            "val_each": 10000,
            "vitr": 200000,
            "vepoch": 128,
            "iterkey": itrk,  # something makes no sense to start fresh
            "dataloader_cfg": dataloader_cfgs,
            # make sure the unseen characters are unseen.
            "modules": modules,
            "routine_cfgs": routines,
            "tasks": task_dict,
        }

def get_mth1000_dan_single_model_train_cfg(save_root, dsroot, log_path, log_each, maxT = 40, itrk="Top Nep",  bsize=48, tvitr=200000,qhb_aug=False, capacity = 512, feat_ch = 512):
    name = "mth1000"
    train_dss = get_mth1000_train_dss(dsroot, maxT, bsize, qhb_aug=qhb_aug)
    res = _get_base_dan_single_model_train_cfg(f"base_{name}_", f"dan_{name}_", train_dss, save_root, log_path, log_each, maxT = maxT, itrk=itrk, bsize=bsize, tvitr=tvitr,capacity=capacity,feat_ch=feat_ch )
    return res
    
def get_mth1200_dan_single_model_train_cfg(save_root, dsroot, log_path, log_each, maxT = 40, itrk="Top Nep",  bsize=48, tvitr=200000,qhb_aug=False, capacity = 512, feat_ch = 512):
    name = "mth1200"
    train_dss = get_mth1200_train_dss(dsroot, maxT, bsize, qhb_aug=qhb_aug)
    res = _get_base_dan_single_model_train_cfg(f"base_{name}_", f"dan_{name}_", train_dss, save_root, log_path, log_each, maxT = maxT, itrk=itrk, bsize=bsize, tvitr=tvitr,capacity=capacity,feat_ch=feat_ch)
    return res

def get_tkh_dan_single_model_train_cfg(save_root, dsroot, log_path, log_each, maxT = 40, itrk="Top Nep",  bsize=48, tvitr=200000,qhb_aug=False, capacity = 512, feat_ch = 512):
    name = "tkh"
    train_dss = get_tkh_train_dss(dsroot, maxT, bsize, qhb_aug=qhb_aug)
    res = _get_base_dan_single_model_train_cfg(f"base_{name}_", f"dan_{name}_", train_dss, save_root, log_path, log_each, maxT = maxT, itrk=itrk, bsize=bsize, tvitr=tvitr,capacity=capacity,feat_ch=feat_ch)
    return res

def get_tkhmth2200_dan_single_model_train_cfg(save_root, dsroot, log_path, log_each, maxT = 40, itrk="Top Nep",  bsize=48, tvitr=200000,qhb_aug=False, capacity = 512, feat_ch = 512):
    name = "tkhmth2200"
    train_dss = get_tkhmth2200_train_dss(dsroot, maxT, bsize, qhb_aug=qhb_aug)
    res = _get_base_dan_single_model_train_cfg(f"base_{name}_", f"dan_{name}_", train_dss, save_root, log_path, log_each, maxT = maxT, itrk=itrk, bsize=bsize, tvitr=tvitr,capacity=capacity,feat_ch=feat_ch)
    return res