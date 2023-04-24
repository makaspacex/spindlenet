import os

from neko_2021_mjt.configs.loadouts.mk8.base_mk8_module_set import arm_base_mk8_task_default
from neko_2021_mjt.configs.routines.ocr_routines.mk8.osdanmk8_routine_cfg import osdanmk8_eval_routine_cfg
from neko_2021_mjt.dss_presets.dual_chhwctw_32 import get_eval_dss
from .configs import model_mod_cfg as modcfg


def dan_mjst_eval_cfg(save_root, dsroot, log_path, iterkey, maxT=30):
    if (log_path):
        epath = os.path.join(log_path, "closeset_benchmarks")
    else:
        epath = None
    te_meta_path_hwdb, te_meta_path_ctw, hwdb_eval_ds, ctw_eval_ds = get_eval_dss(dsroot)
    task_dict = {}
    task_dict = arm_base_mk8_task_default(task_dict,
                                          "base_hwdb_",
                                          osdanmk8_eval_routine_cfg, 1,
                                          te_meta_path_hwdb,
                                          hwdb_eval_ds,
                                          log_path, force_skip_ctx=True)

    return {
        "root": save_root,
        "iterkey": iterkey,  # something makes no sense to start fresh
        "modules": modcfg(None, None),
        "export_path": epath,
        "tasks": task_dict
    }