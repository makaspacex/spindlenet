from .eval_configs import dan_mjst_eval_cfg
from data_root import find_export_root,find_model_root

def main(only_conf=False):
    import sys
    if (len(sys.argv) < 2):
        argv = ["Meeeeooooowwww",
                find_export_root() + "/DUAL_a_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3/jtrmodels/",
                "_E0",
                find_model_root() + "/DUAL_a_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3/jtrmodels/",
                ]
    else:
        argv = sys.argv;
    from neko_2021_mjt.lanuch_std_test import launchtest
    launchtest(argv, dan_mjst_eval_cfg, only_conf=only_conf)

if __name__ == '__main__':
   main()