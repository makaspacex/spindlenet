from neko_2021_mjt.modulars.default_config import get_default_model
from neko_2021_mjt.modulars.spatial_attention import SpatialAttention, \
    SpatialAttentionMk2, spatial_attention_mk3, SpatialAttentionMk5


def get_sa(arg_dict, path, optim_path=None):
    args = {"ifc": arg_dict["num_channels"]}
    # args["num_channels"]=arg_dict["num_channels"]
    return get_default_model(SpatialAttention, args, path, arg_dict["with_optim"], optim_path)


def config_sa(feat_ch=512):
    return \
        {
            "modular": get_sa,
            "save_each": 20000,
            "args":
                {
                    "with_optim": True,
                    "num_channels": feat_ch,
                },
        }


def get_sa_mk2(arg_dict, path, optim_path=None):
    args = {"ifc": arg_dict["num_channels"]}
    # args["num_channels"]=arg_dict["num_channels"]
    return get_default_model(SpatialAttentionMk2, args, path, arg_dict["with_optim"], optim_path)


def config_sa_mk2(feat_ch=512):
    return \
        {
            "modular": get_sa_mk2,
            "save_each": 20000,
            "args":
                {
                    "with_optim": True,
                    "num_channels": feat_ch,
                },
        }


def get_sa_mk3(arg_dict, path, optim_path=None):
    args = {"ifc": arg_dict["num_channels"]}
    # args["num_channels"]=arg_dict["num_channels"]
    return get_default_model(spatial_attention_mk3, args, path, arg_dict["with_optim"], optim_path)


def config_sa_mk3(feat_ch=512):
    return \
        {
            "modular": get_sa_mk3,
            "save_each": 20000,
            "args":
                {
                    "with_optim": True,
                    "num_channels": feat_ch,
                },
        }


def get_sa_mk5(arg_dict, path, optim_path=None):
    args = {"ifc": arg_dict["num_channels"]}
    # args["num_channels"]=arg_dict["num_channels"]
    return get_default_model(SpatialAttentionMk5, args, path, arg_dict["with_optim"], optim_path)


def config_sa_mk5(feat_ch=512):
    return \
        {
            "modular": get_sa_mk5,
            "save_each": 20000,
            "args":
                {
                    "with_optim": True,
                    "num_channels": feat_ch,
                },
        }
