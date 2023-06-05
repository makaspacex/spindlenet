import os

from neko_2021_mjt.configs.loadouts.base_module_set import arm_base_task_default2
from neko_2021_mjt.configs.routines.ocr_routines.mk7.osdanmk7_routine_cfg import osdanmk7_eval_routine_cfg
from neko_2021_mjt.dss_presets.dual_mth_32 import get_mth1200_eval_dss
from configs import model_mod_cfg as modcfg

def dan_mjst_eval_cfg(save_root, dsroot, log_path, iterkey, maxT=30, batchsize=128):
    if (log_path):
        epath = os.path.join(log_path, "closeset_benchmarks")
    else:
        epath = None
    
    maxT = 40
    te_meta_path, eval_ds = get_mth1200_eval_dss(dsroot, maxT,batchsize=batchsize)
    task_dict = {}
    task_dict = arm_base_task_default2(task_dict, "base_mth1200_", osdanmk7_eval_routine_cfg, maxT,
                                       te_meta_path, eval_ds,
                                       log_path)

    return {
        "root": save_root,
        "iterkey": iterkey,  # something makes no sense to start fresh
        "modules": modcfg(te_meta_path, maxT),
        "export_path": epath,
        "tasks": task_dict
    }
