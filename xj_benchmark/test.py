import argparse

import torch

from neko_2021_mjt.neko_abstract_jtr import NekoAbstractModularJointEval
from xj_benchmark.config import get_profile


def get_opts():
    pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskname", type=str, default="DUAL_a_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3")
    opt = parser.parse_args()

    return opt

def launchtest(modcfg_dict, itr_override=None, miter=10000, rot=0, auf=True, maxT_overrider=None):
    if (itr_override is not None):
        itk = itr_override
    trainer = NekoAbstractModularJointEval(modcfg_dict, miter)
    if not auf:
        trainer.modular_dict["pred"].model.UNK_SCR = torch.nn.Parameter(torch.ones_like(trainer.modular_dict["pred"].model.UNK_SCR) * -6000000)
    trainer.val(9, 9, rot)


if __name__ == '__main__':
    opt = get_opts()
    name = opt.taskname
    # name = "DUAL_a_Odancukmk7hdtfnp_r45_C_trinorm_dsa3"
    modcfg_dict = get_profile(name)

    launchtest(modcfg_dict, itr_override=None, miter=10000, rot=0, auf=True, maxT_overrider=None)