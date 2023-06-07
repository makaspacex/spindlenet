from neko_2021_mjt.bogo_modules.res45_binorm_bogomod import NekoRes45BinormBogo


def config_bogo_resbinorm(container, bogoname):
    return {
        "bogo_mod": NekoRes45BinormBogo,
        "args":
            {
                "container": container,
                "name": bogoname,
            }
    }
