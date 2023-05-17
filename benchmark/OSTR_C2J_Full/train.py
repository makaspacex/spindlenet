import sys

from configs import dan_single_model_train_cfg
from neko_2021_mjt.neko_abstract_jtr import NekoModularJointTrainingSemipara
from data_root import find_data_root
import yaml
from data_root import BASE_DIR
from dataloaders.neko_joint_loader import NekoJointLoader

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        bs = int(sys.argv[1])
    else:
        bs = 48
    
    cfgs = dan_single_model_train_cfg(
            "jtrmodels",
            find_data_root(),
            "../logs/",
            200,
            bsize=bs,
            itrk="Top Nep"
        )
    save_conf_path = f"{BASE_DIR}/exp/train/OSTR_C2J_Full.yaml"
    print(f"dumpping file {save_conf_path}")
    yaml.dump(cfgs, open(save_conf_path, 'w+'))
    
    trainer = NekoModularJointTrainingSemipara(cfgs=cfgs)
    # with torch.autograd.detect_anomaly():
    trainer.train(None)
                   