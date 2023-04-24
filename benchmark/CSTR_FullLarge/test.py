from eval_configs import dan_mjst_eval_cfg
from data_root import find_model_root

if __name__ == '__main__':
    import sys

    if (len(sys.argv) < 2):
        argv = ["Meeeeooooowwww",
                None,  # ,find_export_root()+"/CSTR_FullLarge/jtrmodels",
                "_E3",
                find_model_root() + "/CSTR_FullLarge/jtrmodels",
                ]
    else:
        argv = sys.argv
    from neko_2021_mjt.lanuch_std_test import launchtest

    launchtest(argv, dan_mjst_eval_cfg)