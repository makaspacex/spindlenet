from neko_2021_mjt.modulars.dan.classifiers.neko_oslin import NekoOpensetLinearClassifierK
from neko_2021_mjt.modulars.default_config import get_default_model


def get_link_xos(arg_dict, path, optim_path=None):
    args = {
    }
    return get_default_model(NekoOpensetLinearClassifierK, args, path, arg_dict["with_optim"], optim_path)


def config_linxos():
    return \
        {
            "save_each": 20000,
            "modular": get_link_xos,
            "args":
                {
                    "with_optim": True
                },
        }
