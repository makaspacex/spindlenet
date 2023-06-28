import os

from neko_2021_mjt.configs.loadouts.base_module_set import arm_base_task_default2
from neko_2021_mjt.configs.routines.ocr_routines.mk7.osdanmk7_routine_cfg import (
    osdanmk7_eval_routine_cfg,
)
from neko_2021_mjt.dss_presets.dual_mth_32 import get_tkhmth2200_eval_dss
from neko_2021_mjt.dss_presets import dual_mth_32
from configs import model_mod_cfg as modcfg


def dan_mjst_eval_cfg(save_root, dsroot, log_path, iterkey, maxT=40, batchsize=32, data_set_name="tkhmth2200"):
    if log_path:
        epath = os.path.join(log_path, "closeset_benchmarks")
    else:
        epath = None
    
    maxT = 40
    eval_dss_func = getattr(dual_mth_32, f"get_{data_set_name}_eval_dss")
    # te_meta_path, eval_ds = get_tkhmth2200_eval_dss(dsroot, maxT, batchsize=batchsize)
    te_meta_path, eval_ds = eval_dss_func(dsroot, maxT, batchsize=batchsize)
    task_dict = {}
    task_dict = arm_base_task_default2(
        task_dict,
        "base_mth1200_",
        osdanmk7_eval_routine_cfg,
        maxT,
        te_meta_path,
        eval_ds,
        log_path,
    )

    return {
        "root": save_root,
        "iterkey": iterkey,  # something makes no sense to start fresh
        "modules": modcfg(te_meta_path, maxT),
        "export_path": epath,
        "tasks": task_dict,
    }
