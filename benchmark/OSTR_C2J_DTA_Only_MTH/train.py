import sys
from neko_2021_mjt.neko_abstract_jtr import NekoModularJointTrainingSemipara
from data_root import find_data_root
import yaml
from configs import *
import argparse
import configs


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_base", default="/home/izhangxm/mnt/hotssd/vsdf/runtime/OSTR_C2J_DTA_Only_MTH")
    parser.add_argument("--dataset_name", default="tkhmth2200")
    parser.add_argument("--save_name", default=None)
    parser.add_argument("--log_each", default=200)
    parser.add_argument("--bsize", type=int, default=32)
    parser.add_argument("--force", action="store_true", default=False)

    opt = parser.parse_args()
    
    return opt

if __name__ == '__main__':
    
    opt = get_opt()
    dsroot =  find_data_root()
    save_base = opt.save_base
    
    train_cfg_func = getattr(configs, f"get_{opt.dataset_name}_dan_single_model_train_cfg")
    print(train_cfg_func.__name__)
    
    save_name = opt.save_name
    if save_name is None:
        save_name = opt.dataset_name
    
    save_root = f"{save_base}/{save_name}/jtrmodels"
    log_path = f"{save_base}/{save_name}/logs/"
    
    if os.path.exists(save_root) and opt.force == False:
        raise Exception(f"{save_root} is existed.")

    cfgs = train_cfg_func( save_root=save_root,
            dsroot = dsroot,
            log_path=log_path,
            log_each = opt.log_each,
            bsize=opt.bsize,
        )
    os.makedirs(save_root, exist_ok=True)
    save_conf_path = f"{save_root}/OSTR_C2J_DTA_Only_MTH.yaml"
    print(f"dumpping file {save_conf_path}")
    yaml.dump(cfgs, open(save_conf_path, 'w+'))
    
    
    trainer = NekoModularJointTrainingSemipara(cfgs=cfgs)
    trainer.train(None)
