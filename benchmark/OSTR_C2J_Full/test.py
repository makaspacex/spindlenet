from eval_configs import dan_mjst_eval_cfg
from data_root import find_export_root, find_model_root

if __name__ == '__main__':
    import sys

    if (len(sys.argv) < 2):
        argv = ["Meeeeooooowwww",
                "runtime/OSTR_C2J_DTA_Only_MTH/mth_1200/logs/",
                "_E0_I20000",
                "runtime/OSTR_C2J_DTA_Only_MTH/mth_1200/jtrmodels/",
                ]
    else:
        argv = sys.argv
    from neko_2021_mjt.lanuch_std_test import launchtest

    launchtest(argv, dan_mjst_eval_cfg, only_conf=False)
