from eval_configs import dan_mjst_eval_cfg
from data_root import find_export_root, find_model_root
import sys

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        argv = ["Meeeeooooowwww",
                find_export_root() + "/runtime/OSTR_C2J_DTA_Only_MTH/mth_1200_new/logs_E0_test_val/",
                "_E0",
                "runtime/OSTR_C2J_DTA_Only_MTH/mth_1200_new/jtrmodels/",
                ]
        # argv = ["Meeeeooooowwww",
        #         None,
        #         "_E1",
        #         "runtime/OSTR_C2J_DTA_Only_MTH/mth_1200_new/jtrmodels/",
        #         ]
        # Total ED
    else:
        argv = sys.argv
    from neko_2021_mjt.lanuch_std_test import launchtest
    launchtest(argv, dan_mjst_eval_cfg, only_conf=False)
