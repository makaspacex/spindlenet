import sys
from neko_2021_mjt.neko_abstract_jtr import NekoModularJointTrainingSemipara
from data_root import find_data_root
import yaml
from configs import *

if __name__ == '__main__':
    dsroot =  find_data_root()
    save_base = "runtime/OSTR_C2J_DTA_Only_MTH"
    
    # train_cfg_func = get_mth1000_dan_single_model_train_cfg
    train_cfg_func = get_mth1200_dan_single_model_train_cfg
    
    save_root = f"{save_base}/mth_1200_debug/jtrmodels"
    log_path = f"{save_base}/mth_1200_debug/logs/",
    log_each = 200
    bsize=32
    
    cfgs = train_cfg_func( save_root=save_root,
            dsroot = dsroot,
            log_path=log_path,
            log_each = log_each,
            bsize=bsize,
        )
    
    save_conf_path = f"runtime/OSTR_C2J_DTA_Only_MTH_debug.yaml"
    print(f"dumpping file {save_conf_path}")
    yaml.dump(cfgs, open(save_conf_path, 'w+'))
    
    trainer = NekoModularJointTrainingSemipara(cfgs=cfgs)
    trainer.train(None)
