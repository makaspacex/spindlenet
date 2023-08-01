from eval_configs import dan_mjst_eval_cfg
from data_root import find_export_root, find_model_root
import sys
from neko_2021_mjt.lanuch_std_test import launchtest
import yaml
import os
from data_root import find_data_root
    
if __name__ == "__main__":
    iter_k = "_E1"
    iter_k = "latest"
    data_set_name = "tkhmth2200"
    data_set_name = "mth1200"
    data_set_name = "tkh"
    data_set_name = "mth1000"
    exp_name="aaa"
    
    model_root =  f"/home/izhangxm/mnt/hotssd/vsdf/runtime/OSTR_C2J_DTA_Only_MTH/{exp_name}/jtrmodels/"
    export_path = f"runtime/OSTR_C2J_DTA_Only_MTH/{exp_name}/logs_test_{iter_k}/"
    
    modscc = dan_mjst_eval_cfg(
            model_root,
            find_data_root(),
            export_path,
            iter_k,
            data_set_name=data_set_name
        )
    
    export_path=modscc.get("export_path", export_path)
    os.makedirs(export_path, exist_ok=True)
    save_conf_path = os.path.join(export_path, f"{data_set_name}.yaml")
    print(f"dumpping file {save_conf_path}")
    yaml.dump(modscc, open(save_conf_path, 'w+'))
    # exit(0)
    
    launchtest(sys.argv, modscc, only_conf=False)
