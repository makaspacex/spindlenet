from eval_configs import dan_mjst_eval_cfg
from data_root import find_export_root, find_model_root
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        argv = [
            "Meeeeooooowwww",
             "runtime/OSTR_C2J_DTA_Only_MTH/tkhmth2200_v1/logs_E1_tkhmth2200_test/",
            "_E1",
            "/home/izhangxm/Desktop/work/main/VSDF/runtime/OSTR_C2J_DTA_Only_MTH/tkhmth2200_v1/jtrmodels/",
        ]
        # argv = ["Meeeeooooowwww",
        #         None,
        #         "_E1",
        #         "runtime/OSTR_C2J_DTA_Only_MTH/mth_1200_new/jtrmodels/",
        #         ]
        # Total ED
        # 统计指标，如果长度对不上，就忽略这个词条。
        # 长度的关系
        # 错误的字符频率和比例
    else:
        argv = sys.argv
    from neko_2021_mjt.lanuch_std_test import launchtest

    launchtest(argv, dan_mjst_eval_cfg, only_conf=False)
