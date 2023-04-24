from eval_configs import dan_mjst_eval_cfg
from data_root import find_model_root

if __name__ == '__main__':
    import sys

    for i in ["500", "1000", "1500", "2000"]:
        print("-----------------", i, "starts")
        if (len(sys.argv) < 2):
            argv = ["Meeeeooooowwww",
                    None,  # find_export_root()+"ZSCR_Handwritten_Full/jtrmodels"+i+"/",
                    "_E4",
                    find_model_root() + "ZSCR_Handwritten_Full/jtrmodels" + i + "/",
                    ]
        else:
            argv = sys.argv
        from neko_2021_mjt.lanuch_std_test import launchtest

        launchtest(argv, dan_mjst_eval_cfg)
        print("-----------------", i, "ends")
