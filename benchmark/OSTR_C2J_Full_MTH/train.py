import sys

from configs import dan_single_model_train_cfg_for_mth
from neko_2021_mjt.neko_abstract_jtr import NekoModularJointTrainingSemipara
from data_root import find_data_root
import yaml
from dataloaders.neko_joint_loader import NekoJointLoader

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        bs = int(sys.argv[1])
    else:
        bs = 32
    
    cfgs = dan_single_model_train_cfg_for_mth(
            "runtime/jtrmodels",
            find_data_root(),
            "runtime/logs/",
            200,
            bsize=bs,
            itrk="Top Nep"
        )
    # save_conf_path = f"{BASE_DIR}/exp/train/OSTR_C2J_Full_MTH.yaml"
    # print(f"dumpping file {save_conf_path}")
    # yaml.dump(cfgs, open(save_conf_path, 'w+'))
    
    trainer = NekoModularJointTrainingSemipara(cfgs=cfgs)
    trainer.train(None, flag=True)