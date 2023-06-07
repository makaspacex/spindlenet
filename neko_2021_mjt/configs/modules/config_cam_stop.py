from neko_2020nocr.dan.configs.pipelines_pami import get_cam_args
from neko_2021_mjt.modulars.dan.CAM_stop import NekoCAMStop

from neko_2021_mjt.modulars.default_config import get_default_model


def get_cam_stop(arg_dict, path, optim_path=None):
    args = get_cam_args(arg_dict["maxT"], arg_dict["cam_ch"])
    args["scales"] = arg_dict["scales"]
    # args["num_channels"]=arg_dict["num_channels"]
    return get_default_model(NekoCAMStop, args, path, arg_dict["with_optim"], optim_path)


def config_cam_stop(maxT, scales=None, feat_ch=512, expf=1, cam_ch=64):
    if scales is None:
        scales = [
            [int(expf * 64), 16, 64],
            [int(expf * 256), 8, 32],
            [int(feat_ch), 8, 32]
        ]
    print(scales)
    return \
        {
            "modular": get_cam_stop,
            "save_each": 20000,
            "args":
                {
                    "cam_ch": cam_ch,
                    "num_channels": feat_ch,
                    "scales": scales,
                    "maxT": maxT,
                    "with_optim": True
                },
        }
