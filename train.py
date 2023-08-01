from omegaconf import DictConfig, OmegaConf
import hydra
from neko_2021_mjt.neko_abstract_jtr import NekoModularJointTrainingSemipara

print(hydra.__version__)

@hydra.main(config_path='experiments', version_base="1.3")
def main(cfg: DictConfig) -> None:
    # OmegaConf.resolve(cfg)
    # print(OmegaConf.to_yaml(cfg))
    if len(cfg) == 0:
        raise Exception("you must specific a config name")
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    
    trainer = NekoModularJointTrainingSemipara(cfgs= OmegaConf.to_object(cfg))
    trainer.train(None, s_e=cfg.se, val_epoch=cfg.val_epoch)

if __name__ == "__main__":
    main()
