from neko_2021_mjt.configs.loadouts.base_module_set import arm_base_routine2
from neko_2021_mjt.configs.loadouts.base_module_set import arm_base_task_default2
from neko_2021_mjt.configs.loadouts.mk7.lsctt_module_set_mk7 import arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7
from neko_2021_mjt.configs.routines.ocr_routines.mk7.osdanmk7_routine_cfg import osdanmk7_eval_routine_cfg
from neko_2021_mjt.configs.routines.ocr_routines.mk7.osdanmk7_routine_cfg import osdanmk7dt_ocr_routine
from neko_2021_mjt.dss_presets.dual_no_lsct_32 import get_dss
from neko_2021_mjt.dss_presets.dual_mth_32 import get_mth1200_train_dss

def model_mod_cfg(tr_meta_path_chs, maxT_chs):
    capacity = 256 * 4
    feat_ch = 512
    # mods=arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7(mods,"base_mjst_",maxT_mjst,capacity,feat_ch,tr_meta_path_mjst,wemb=0)
    mods = arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7(prefix="base_mth1200_", maxT=maxT_chs, capacity=capacity, feat_ch=feat_ch, tr_meta_path=tr_meta_path_chs, srcdst=None, wemb=0)
    return mods

def dan_single_model_train_cfg(save_root, dsroot, log_path, log_each, itrk="Top Nep", bsize=48, tvitr=200000):
    maxT = 40
    tr_meta_path, train_joint_ds, eval_ds, te_meta_path= get_mth1200_train_dss(dsroot, maxT, bsize)

    task_dict = {}
    task_dict = arm_base_task_default2(task_dict, "base_mth1200_", 
                                       osdanmk7_eval_routine_cfg, 
                                       maxT, 
                                       tr_meta_path,
                                       eval_ds,
                                       log_path)

    routines = {}
    routines = arm_base_routine2(routines, "base_mth1200_", osdanmk7dt_ocr_routine, maxT, log_path,log_each, "dan_mth1200_")

    return \
        {
            "root": save_root,
            "val_each": 10000,
            "vitr": 200000,
            "vepoch": 2,
            "iterkey": itrk,  # something makes no sense to start fresh
            "dataloader_cfg": train_joint_ds,
            # make sure the unseen characters are unseen.
            "modules": model_mod_cfg(tr_meta_path, maxT),
            "routine_cfgs": routines,
            "tasks": task_dict,
        }
