from neko_2021_mjt.configs.data.mjst_data import get_mjstcqa_cfg
from neko_2021_mjt.configs.data.mjst_data_asize import get_mjstcqaAS_cfg
from dataloaders import neko_joint_loader


def get_mjstcqa_dataloadercfgs(root, maxT, bs=48, hw=[32, 128]):
    return \
        {
            "loadertype": neko_joint_loader,
            "subsets":
                {
                    "dan_mjstcqa": get_mjstcqa_cfg(root, maxT, bs=bs, hw=hw)
                }
        }


def get_mjstcqa_as_dataloadercfgs(root, maxT):
    return \
        {
            "loadertype": neko_joint_loader,
            "subsets":
                {
                    "dan_mjstcqa": get_mjstcqaAS_cfg(root, maxT)
                }
        }
