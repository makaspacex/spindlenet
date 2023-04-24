from eval_configs import dan_mjst_eval_cfg
from data_root import find_export_root, find_model_root
import sys

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        argv = ["Meeeeooooowwww",
                find_export_root() + "/OSTR_C2J_DTAOnly/jtrmodels/",
                "_E0",
                find_model_root() + "/OSTR_C2J_DTAOnly/jtrmodels/",
                ]
    else:
        argv = sys.argv
    from neko_2021_mjt.lanuch_std_test import launchtest

    launchtest(argv, dan_mjst_eval_cfg)