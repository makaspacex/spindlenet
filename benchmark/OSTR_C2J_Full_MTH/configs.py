from neko_2021_mjt.configs.loadouts.mk8.base_mk8_module_set import arm_base_mk8_routine, arm_base_mk8_task_default
from neko_2021_mjt.configs.loadouts.mk8.base_mk8_module_set import arm_trinorm_mk8hnp_module_set_dan_r45
from neko_2021_mjt.configs.routines.ocr_routines.mk8.osdanmk8_routine_cfg import osdanmk8_eval_routine_cfg
from neko_2021_mjt.configs.routines.ocr_routines.mk8.osdanmk8_routine_cfg import osdanmk8adt_ocr_routine
from neko_2021_mjt.dss_presets.dual_mth_32 import get_mth1200_train_dss


def model_mod_cfg(tr_meta_path, maxT_chs):
    capacity = 256
    feat_ch = 512
    mods = {}
    mods = arm_trinorm_mk8hnp_module_set_dan_r45(mods, "base_mth1200_", maxT_chs, capacity, feat_ch, tr_meta_path,
                                                 ccnt=7000, wemb=0)
    return mods

def dan_single_model_train_cfg_for_mth(save_root, dsroot, log_path, log_each, itrk="Top Nep", bsize=48, tvitr=200000):
    maxT = 40

    tr_meta_path, train_joint_ds,eval_ds,te_meta_path = get_mth1200_train_dss(dsroot, maxT, bsize)

    task_dict = {}
    task_dict = arm_base_mk8_task_default(task_dict, "base_mth1200_", osdanmk8_eval_routine_cfg, maxT,
                                          te_meta_path, eval_ds,
                                          log_path, force_skip_ctx=True)

    routines = {}
    routines = arm_base_mk8_routine(routines, "base_mth1200_", osdanmk8adt_ocr_routine, maxT, log_path, log_each,
                                    "dan_mth1200_", view_name="synthw", proto_viewname="glyph")

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
