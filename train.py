from omegaconf import DictConfig, OmegaConf
import hydra
from neko_2021_mjt.neko_abstract_jtr import NekoModularJointTrainingSemipara

print(hydra.__version__)

def _work(cfg: DictConfig):
    trainer = NekoModularJointTrainingSemipara(cfgs= OmegaConf.to_object(cfg))
    trainer.train(None, s_e=cfg.se, val_epoch=cfg.val_epoch)

@hydra.main(config_path='experiments', config_name="spindle_ori", version_base="1.3")
def spindle_ori(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    _work(cfg=cfg)
    
@hydra.main(config_path='experiments', config_name="spindle_v1", version_base="1.3")
def spindle_v1(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    _work(cfg=cfg)

@hydra.main(config_path='experiments', config_name="spindle_v2", version_base="1.3")
def spindle_v2(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    _work(cfg=cfg)

@hydra.main(config_path='experiments', config_name="spindle_v3", version_base="1.3")
def spindle_v3(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    _work(cfg=cfg)

@hydra.main(config_path='experiments', config_name="spindle_v4", version_base="1.3")
def spindle_v4(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    _work(cfg=cfg)

if __name__ == "__main__":
    # spindle_ori()
    # spindle_v1()
    # spindle_v2()
    # spindle_v3()
    spindle_v4()


