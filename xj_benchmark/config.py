import os

from data_root import find_data_root
from data_root import find_export_root, find_model_root
from neko_2021_mjt.configs.loadouts.base_module_set import arm_base_task_default2
from neko_2021_mjt.configs.loadouts.mk7.lsctt_module_set_mk7 import arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7
from neko_2021_mjt.configs.loadouts.mk8.base_mk8_module_set import arm_base_mk8_task_default
from neko_2021_mjt.configs.loadouts.mk8.base_mk8_module_set import arm_trinorm_mk8hnp_module_set_dan_r45
from neko_2021_mjt.configs.loadouts.mk8.base_mk8_module_set import arm_trinorm_mk8hnp_module_set_dan_r45ptpt
from neko_2021_mjt.configs.routines.ocr_routines.mk7.osdanmk7_routine_cfg import osdanmk7_eval_routine_cfg
from neko_2021_mjt.configs.routines.ocr_routines.mk8.osdanmk8_routine_cfg import osdanmk8_eval_routine_cfg
from neko_2021_mjt.dss_presets.dual_chhwctw_32 import get_eval_dss as get_eval_dss2
from neko_2021_mjt.dss_presets.dual_no_lsct_32 import get_eval_dss

def model_mod_cfg(tr_meta_path_chs, tr_meta_path_mjst, maxT_mjst, maxT_chs):
    capacity = 256
    feat_ch = 512
    mods = {}
    mods = arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7(mods, "base_chs_", maxT_chs, capacity, feat_ch,tr_meta_path_chs, wemb=0)
    return mods

def model_mod_cfg3(tr_meta_path_chs, tr_meta_path_mjst, maxT_mjst, maxT_chs):
    capacity = 256
    feat_ch = 512
    mods = {}
    mods = arm_trinorm_mk8hnp_module_set_dan_r45(mods, "base_chs_", maxT_chs, capacity, feat_ch, tr_meta_path_chs, ccnt=3824, wemb=0)
    return mods


def model_mod_cfg4(tr_meta_path_chs, tr_meta_path_mjst, maxT_mjst, maxT_chs):
    capacity = 256
    feat_ch = 512
    mods = {}
    mods = arm_trinorm_mk8hnp_module_set_dan_r45ptpt(mods, "base_chs_", maxT_chs, capacity, feat_ch, tr_meta_path_chs, ccnt=3824, wemb=0, expf=1.5)
    return mods


def model_mod_cfg5(tr_meta_path_chs, tr_meta_path_mjst, maxT_mjst, maxT_chs):
    capacity = 256
    feat_ch = 512
    mods = {}
    mods = arm_trinorm_mk8hnp_module_set_dan_r45ptpt(mods, "base_mjst_", maxT_mjst, capacity, feat_ch, tr_meta_path_mjst, ccnt=38, wemb=0, expf=1.5)
    return mods


def model_mod_cfg6(tr_meta_path_ctw, tr_meta_path_hwdb):
    capacity = 256
    feat_ch = 512
    mods = {}

    mods = arm_trinorm_mk8hnp_module_set_dan_r45(mods, "base_ctw_", 1, capacity, feat_ch, tr_meta_path_ctw, wemb=0)
    return mods


def model_mod_cfg7(tr_meta_path_ctw, tr_meta_path_hwdb):
    capacity = 256
    feat_ch = 512
    mods = {}
    mods = arm_trinorm_mk8hnp_module_set_dan_r45(mods, "base_hwdb_", 1, capacity, feat_ch, tr_meta_path_hwdb, wemb=0)
    return mods


def get_argv(name):
    export_root = find_export_root() + f"/{name}/jtrmodels/"
    model_root = find_model_root() + f"/{name}/jtrmodels/"
    argv = ["Meeeeooooowwww", export_root, "_E0", model_root]
    return argv


def get_modcfg_dict(name, model_root, datasets_root, export_root, iterkey):

    epath = os.path.join(export_root, "benchmarks")

    task_dict = {}
    log_path = export_root

    if name in ["DUAL_a_Odancukmk7hdtfnp_r45_C_trinorm_dsa3", "DUAL_a_Odancukmk7hnp_r45_C_trinorm_dsa3"]:
        te_meta_path_chsjap, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds = get_eval_dss(datasets_root, 25, 30)
        task_dict = arm_base_task_default2(task_dict, "base_chs_", osdanmk7_eval_routine_cfg, 30, te_meta_path_chsjap,  chs_eval_ds, log_path)
        return {
            "root": model_root,
            "iterkey": iterkey,  # something makes no sense to start fresh
            "modules": model_mod_cfg(None, None, 25, 30),
            "export_path": epath,
            "tasks": task_dict
        }
    elif name == "DUAL_a_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3":
        te_meta_path_chsjap, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds = get_eval_dss(datasets_root, 25, 30)
        task_dict = {}
        task_dict = arm_base_mk8_task_default(task_dict, "base_chs_", osdanmk8_eval_routine_cfg, 30,
                                              te_meta_path_chsjap, chs_eval_ds,
                                              log_path, force_skip_ctx=True)

        return {
            "root": model_root,
            "iterkey": iterkey,  # something makes no sense to start fresh
            "modules": model_mod_cfg3(None, None, 25, 30),
            "export_path": epath,
            "tasks": task_dict
        }
    elif name == "DUAL_a_Odancukmk8ahdtfnp_r45pttpt_C_trinorm_dsa3":
        te_meta_path_chsjap, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds = get_eval_dss(datasets_root, 25, 30)
        task_dict = {}
        task_dict = arm_base_mk8_task_default(task_dict, "base_chs_", osdanmk8_eval_routine_cfg, 30,
                                              te_meta_path_chsjap, chs_eval_ds,
                                              log_path, force_skip_ctx=True)
        return {
            "root": model_root,
            "iterkey": iterkey,  # something makes no sense to start fresh
            "modules": model_mod_cfg4(None, None, 25, 30),
            "export_path": epath,
            "tasks": task_dict
        }
    elif name == "DUAL_b_Odancukmk8ahdtfnp_r45pttpt_C_trinorm_dsa3":
        te_meta_path_chsjap, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds = get_eval_dss(datasets_root, 25, 30)
        task_dict = {}
        task_dict = arm_base_mk8_task_default(task_dict, "base_chs_", osdanmk8_eval_routine_cfg, 30,
                                              te_meta_path_chsjap, chs_eval_ds,
                                              log_path, force_skip_ctx=True)

        return {
            "root": model_root,
            "iterkey": iterkey,  # something makes no sense to start fresh
            "modules": model_mod_cfg5(None, None, 25, 30),
            "export_path": epath,
            "tasks": task_dict
        }
    # 这个比较特殊，和其他的不一样 先暂时搁置
    elif name == "DUAL_ch_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3":

        te_meta_path_hwdb, te_meta_path_ctw, hwdb_eval_ds, ctw_eval_ds = get_eval_dss2(datasets_root)
        task_dict = {}
        task_dict = arm_base_mk8_task_default(task_dict,
                                              "base_ctw_",
                                              osdanmk8_eval_routine_cfg, 1,
                                              te_meta_path_ctw,
                                              ctw_eval_ds,
                                              log_path, force_skip_ctx=True)

        return {
            "root": model_root,
            "iterkey": iterkey,  # something makes no sense to start fresh
            "modules": model_mod_cfg6(None, None),
            "export_path": epath,
            "tasks": task_dict
        }

    # 这个比较特殊，和其他的不一样 先暂时搁置
    elif name == "DUAL_chhw_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3":
        te_meta_path_hwdb, te_meta_path_ctw, hwdb_eval_ds, ctw_eval_ds = get_eval_dss2(datasets_root)
        task_dict = {}
        task_dict = arm_base_mk8_task_default(task_dict,
                                              "base_hwdb_",
                                              osdanmk8_eval_routine_cfg, 1,
                                              te_meta_path_hwdb,
                                              hwdb_eval_ds,
                                              log_path, force_skip_ctx=True)

        return {
            "root": model_root,
            "iterkey": iterkey,  # something makes no sense to start fresh
            "modules": model_mod_cfg7(None, None),
            "export_path": epath,
            "tasks": task_dict
        }

    raise  Exception("未知的名称")

def get_profile(name):
    argv, modcfg_dict = None, None

    export_root = find_export_root() + f"/{name}/jtrmodels/"
    model_root = find_model_root() + f"/{name}/jtrmodels/"

    itk_names = {
        "DUAL_a_Odancukmk7hdtfnp_r45_C_trinorm_dsa3": "_E0",
        "DUAL_a_Odancukmk7hnp_r45_C_trinorm_dsa3": "_E0",
        "DUAL_a_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3": "_E0",
        "DUAL_a_Odancukmk8ahdtfnp_r45pttpt_C_trinorm_dsa3": "_E0",
        "DUAL_b_Odancukmk8ahdtfnp_r45pttpt_C_trinorm_dsa3": "_E3",
        "DUAL_ch_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3": "_E4",
        "DUAL_chhw_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3": "_E4"
         }

    itk = itk_names[name]
    datasets_root =  find_data_root()
    modcfg_dict = get_modcfg_dict(name, model_root, datasets_root, export_root, itk)

    return modcfg_dict


if __name__ == '__main__':
    argv1, modcfg_dict1 = get_profile("DUAL_a_Odancukmk7hdtfnp_r45_C_trinorm_dsa3")
    argv2, modcfg_dict2 = get_profile("DUAL_a_Odancukmk7hnp_r45_C_trinorm_dsa3")
    argv3, modcfg_dict3 = get_profile("DUAL_a_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3")
    argv4, modcfg_dict4 = get_profile("DUAL_a_Odancukmk8ahdtfnp_r45pttpt_C_trinorm_dsa3")
