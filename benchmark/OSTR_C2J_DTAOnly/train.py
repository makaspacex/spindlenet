import sys

from configs import dan_single_model_train_cfg
from neko_2021_mjt.neko_abstract_jtr import NekoModularJointTrainingSemipara
from data_root import find_data_root

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
    
    
    
    
    trainer = NekoModularJointTrainingSemipara(
        
    )
    # with torch.autograd.detect_anomaly():
    trainer.train(None)
