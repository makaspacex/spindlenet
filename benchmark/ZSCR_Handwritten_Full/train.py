import sys

from configs import dan_single_model_train_cfg
from neko_2021_mjt.neko_abstract_jtr import NekoModularJointTrainingPara
from data_root import find_data_root

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        ccnt = int(sys.argv[1])
    else:
        ccnt = 500

    trainer = NekoModularJointTrainingPara(
        dan_single_model_train_cfg(
            "jtrmodels" + str(ccnt),
            find_data_root(),
            ccnt,
            "../logs/",
            200,
        )
    )
    # with torch.autograd.detect_anomaly():
    trainer.train(None)
