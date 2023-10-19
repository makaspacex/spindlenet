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
    
    root = cfg['root']
    
    pths = glob.glob(f"{root}*.pth")
    max_e = -1
    for p in pths:
        # print(p)
        name = os.path.basename(p)
        _r =  re.match(r".*?_E([\d]{1,}).pth", name)
        if _r:
            _n = int(_r.group(1))
            if _n > max_e:
                max_e = _n
            print(_r,name)
    
    iters_keys = [f"_E{x}" for x in range(0, max_e+1)]
    iters_keys += ["lastest"]
    print(iters_keys)
    for index, iterkey in enumerate(iters_keys):
        _cfg = copy.deepcopy(cfg)
        _cfg['export_path'] = None
        _cfg['iterkey'] = iterkey
        
        val_epoch = _cfg.get('val_epoch',1)
        if index % val_epoch != 0:
            continue
        
        OmegaConf.resolve(_cfg)
        print(OmegaConf.to_yaml(_cfg))
        
        _cfg = OmegaConf.to_object(_cfg)
        launchtest({},_cfg , only_conf=False)

if __name__ == "__main__":
    main()
