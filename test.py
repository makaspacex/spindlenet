from omegaconf import DictConfig, OmegaConf
import hydra
from neko_2021_mjt.neko_abstract_jtr import NekoModularJointTrainingSemipara
import sys
from neko_2021_mjt.lanuch_std_test import launchtest
import copy
import glob
import os
import re

print(hydra.__version__)

@hydra.main(config_path='experiments', version_base="1.3")
def main(cfg: DictConfig) -> None:
    # OmegaConf.resolve(cfg)
    # print(OmegaConf.to_yaml(cfg))
    if len(cfg) == 0:
        raise Exception("you must specific a config name")
    
    _cfg = cfg
    OmegaConf.resolve(_cfg)
    print(OmegaConf.to_yaml(_cfg))
    _cfg = OmegaConf.to_object(_cfg)
    
    _cfg['export_path'] = None
    launchtest({},_cfg , only_conf=False)
    

if __name__ == "__main__":
    main()
