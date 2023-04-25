import os

from neko_2021_mjt.configs.loadouts.mk8.base_mk8_module_set import arm_base_mk8_task_default
from neko_2021_mjt.configs.routines.ocr_routines.mk8.osdanmk8_routine_cfg import osdanmk8_eval_routine_cfg
from neko_2021_mjt.dss_presets.dual_no_lsct_32 import get_eval_dss
from configs import model_mod_cfg as modcfg


def dan_mjst_eval_cfg(save_root, dsroot, log_path, iterkey, maxT=30):
    if (log_path):
        epath = os.path.join(log_path, "closeset_benchmarks")
    else:
        epath = None
    te_meta_path_chsjap, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds = get_eval_dss(dsroot, 25, 30)
    task_dict = {}
    task_dict = arm_base_mk8_task_default(task_dict, "base_chs_", osdanmk8_eval_routine_cfg, 30,
                                          te_meta_path_chsjap, chs_eval_ds,
                                          log_path, force_skip_ctx=True)

    return \
        {
            "root": save_root,
            "iterkey": iterkey,  # something makes no sense to start fresh
            "modules": modcfg(None, None, 25, 30),
            "export_path": epath,
            "tasks": task_dict
        }
