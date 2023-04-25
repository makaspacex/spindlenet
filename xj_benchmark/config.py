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


# def model_mod_cfg(tr_meta_path_chs, tr_meta_path_mjst, maxT_mjst, maxT_chs):
#     capacity = 256
#     feat_ch = 512
#     mods = {}
#     mods = arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7(mods, "base_chs_", maxT_chs, capacity, feat_ch,
#                                                          tr_meta_path_chs, wemb=0)
#     return mods
#

def model_mod_cfg3(tr_meta_path_chs, tr_meta_path_mjst, maxT_mjst, maxT_chs):
    capacity = 256
    feat_ch = 512
    mods = {}
    mods = arm_trinorm_mk8hnp_module_set_dan_r45(mods, "base_chs_", maxT_chs, capacity, feat_ch, tr_meta_path_chs,
                                                 ccnt=3824, wemb=0)
    return mods


def model_mod_cfg4(tr_meta_path_chs, tr_meta_path_mjst, maxT_mjst, maxT_chs):
    capacity = 256
    feat_ch = 512
    mods = {}
    mods = arm_trinorm_mk8hnp_module_set_dan_r45ptpt(mods, "base_chs_", maxT_chs, capacity, feat_ch, tr_meta_path_chs,
                                                     ccnt=3824, wemb=0, expf=1.5)
    return mods


def model_mod_cfg5(tr_meta_path_chs, tr_meta_path_mjst, maxT_mjst, maxT_chs):
    capacity = 256
    feat_ch = 512
    mods = {}
    mods = arm_trinorm_mk8hnp_module_set_dan_r45ptpt(mods, "base_mjst_", maxT_mjst, capacity, feat_ch,
                                                     tr_meta_path_mjst, ccnt=38, wemb=0, expf=1.5)
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


def get_modcfg_dict(name, model_root, datasets_root, export_root, iterkey):
    epath = os.path.join(export_root, "benchmarks")
    log_path = export_root

    modules = {}
    task_dict = {}

    te_meta_path, _, _, datasets = get_eval_dss(dsroot=datasets_root, maxT_mjst=25, maxT_chs=30)

    if name in ["OSTR_C2J_BaseModel", "OSTR_C2J_DTAOnly"]:
        dconf = {
            "itk": "_E0",
            "prefix": "base_chs_",
            "db_dir_name": "Dual_a_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3",
            "expf": 1,
            "fecnt": 3,
            "name": "close_set_benchmarks",
            #
            "capacity": 256,
            "feat_ch": 512,
            "tr_meta_path": None,
            "maxT": 30,
            "wemb": 0
        }

        task_dict = arm_base_task_default2(task_dict,
                                           prefix=dconf["prefix"],
                                           routine_type=osdanmk7_eval_routine_cfg,
                                           maxT=dconf["maxT"],
                                           te_meta_path=te_meta_path,
                                           datasets=datasets,
                                           log_path=log_path,
                                           name=dconf["name"])

        modules = arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7(prefix=dconf["prefix"],
                                                                maxT=dconf["maxT"],
                                                                capacity=dconf["capacity"],
                                                                feat_ch=dconf["feat_ch"],
                                                                tr_meta_path=dconf["tr_meta_path"],
                                                                wemb=dconf["wemb"])

    elif name == "OSTR_C2J_Full":
        task_dict = arm_base_mk8_task_default(srcdst=task_dict,
                                              prefix="base_chs_",
                                              routine_type=osdanmk8_eval_routine_cfg,
                                              maxT=30,
                                              te_meta_path=te_meta_path,
                                              datasets=datasets,
                                              log_path=log_path,
                                              force_skip_ctx=True)

        return {
            "root": model_root,
            "iterkey": iterkey,  # something makes no sense to start fresh
            "modules": model_mod_cfg3(None, None, 25, 30),
            "export_path": epath,
            "tasks": task_dict
        }


    elif name == "OSTR_C2J_FullLarge":
        task_dict = arm_base_mk8_task_default(task_dict, "base_chs_", osdanmk8_eval_routine_cfg, 30,
                                              te_meta_path, datasets,
                                              log_path, force_skip_ctx=True)
        return {
            "root": model_root,
            "iterkey": iterkey,  # something makes no sense to start fresh
            "modules": model_mod_cfg4(None, None, 25, 30),
            "export_path": epath,
            "tasks": task_dict
        }
    elif name == "CSTR_FullLarge":
        task_dict = {}
        task_dict = arm_base_mk8_task_default(task_dict, "base_mjst_", osdanmk8_eval_routine_cfg, 30,
                                              te_meta_path, datasets,
                                              log_path, force_skip_ctx=True)

        return {
            "root": model_root,
            "iterkey": iterkey,  # something makes no sense to start fresh
            "modules": model_mod_cfg5(None, None, 25, 30),
            "export_path": epath,
            "tasks": task_dict
        }
    # 这个比较特殊，和其他的不一样 先暂时搁置
    elif name == "ZSCR_CTW_Full":

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
    elif name == "ZSCR_Handwritten_Full":
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

    return {
        "root": model_root,
        "iterkey": iterkey,  # something makes no sense to start fresh
        "modules": modules,
        "export_path": epath,
        "tasks": task_dict
    }

    # raise Exception("未知的名称")


def get_profile(name):
    argv, modcfg_dict = None, None

    itk_names = {
        "OSTR_C2J_DTAOnly": "_E0",
        "OSTR_C2J_BaseModel": "_E0",
        "OSTR_C2J_Full": "_E0",
        "OSTR_C2J_FullLarge": "_E0",
        "CSTR_FullLarge": "_E3",
        "ZSCR_CTW_Full": "_E4",
        "ZSCR_Handwritten_Full": "_E4"
    }

    db_dir_dict = {
        "OSTR_C2J_DTAOnly": "DUAL_a_Odancukmk7hdtfnp_r45_C_trinorm_dsa3",
        "OSTR_C2J_BaseModel": "DUAL_a_Odancukmk7hnp_r45_C_trinorm_dsa3",
        "OSTR_C2J_Full": "DUAL_a_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3",
        "OSTR_C2J_FullLarge": "DUAL_a_Odancukmk8ahdtfnp_r45pttpt_C_trinorm_dsa3",
        "CSTR_FullLarge": "DUAL_b_Odancukmk8ahdtfnp_r45pttpt_C_trinorm_dsa3",
        "ZSCR_CTW_Full": "DUAL_ch_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3",
        "ZSCR_Handwritten_Full": "DUAL_chhw_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3"
    }

    export_root = find_export_root() + f"/{db_dir_dict[name]}/jtrmodels/"
    model_root = find_model_root() + f"/{db_dir_dict[name]}/jtrmodels/"


    itk = itk_names[name]
    datasets_root = find_data_root()

    modcfg_dict = get_modcfg_dict(name, model_root, datasets_root, export_root, itk)

    return modcfg_dict

if __name__ == '__main__':
    itk_names = {
        "OSTR_C2J_DTAOnly": "_E0",
        "OSTR_C2J_BaseModel": "_E0",
        "OSTR_C2J_Full": "_E0",
        "OSTR_C2J_FullLarge": "_E0",
        "CSTR_FullLarge": "_E3",
        "ZSCR_CTW_Full": "_E4",
        "ZSCR_Handwritten_Full": "_E4"
    }

    import yaml
    for dd_name, v in itk_names.items():
        aaa = get_profile(dd_name)
        yaml.dump(aaa, open(f"../exp/{dd_name}.yaml", "w+"))

    #
    #
    # modcfg_dict1 = get_profile("OSTR_C2J_DTAOnly")
    # modcfg_dict2 = get_profile("OSTR_C2J_BaseModel")
    # modcfg_dict3 = get_profile("OSTR_C2J_Full")
    # modcfg_dict4 = get_profile("OSTR_C2J_FullLarge")
    # modcfg_dict4 = get_profile("ZSCR_CTW_Full")
    # modcfg_dict4 = get_profile("ZSCR_Handwritten_Full")
