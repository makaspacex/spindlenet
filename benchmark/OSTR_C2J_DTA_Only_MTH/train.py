import sys

from configs import dan_single_model_train_cfg
from neko_2021_mjt.neko_abstract_jtr import NekoModularJointTrainingSemipara
from data_root import find_data_root
import yaml

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        bs = int(sys.argv[1])
    else:
        bs = 32
    
    cfgs = dan_single_model_train_cfg(
            "runtime/OSTR_C2J_DTA_Only_MTH/mth_1200_new_debug/jtrmodels",
            find_data_root(),
            "runtime/OSTR_C2J_DTA_Only_MTH/mth_1200_new_debug/logs/",
            200,
            bsize=bs,
            itrk="Top Nep"
        )
    
    # save_conf_path = f"exp/train/OSTR_C2J_DTA_Only_MTH.yaml"
    # print(f"dumpping file {save_conf_path}")
    # yaml.dump(cfgs, open(save_conf_path, 'w+'))
    
    trainer = NekoModularJointTrainingSemipara(cfgs=cfgs)
    # with torch.autograd.detect_anomaly():
    trainer.train(None)
