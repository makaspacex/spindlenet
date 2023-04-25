import glob
import os
import os.path
import shutil

import torch
from neko_2021_mjt.neko_abstract_jtr import NekoAbstractModularJointEval
from data_root import find_data_root
from data_root import find_export_root


def testready(argv, modcfg, temeta, itr_override=None, miter=10000, rot=0, auf=True, maxT_overrider=None):
    from neko_2021_mjt.neko_abstract_jtr import NekoAbstractModularJointEval
    from data_root import find_data_root
    if (len(argv) > 2):
        export_path = argv[1]
        if (export_path == "None"):
            export_path = None
        itk = argv[2]
        root = argv[3]
    else:
        export_path = find_export_root()
        itk = "latest"
        root = "jtrmodels"
    if (itr_override is not None):
        itk = itr_override
    if (maxT_overrider is None):
        trainer = NekoAbstractModularJointEval(
            modcfg(
                root,
                find_data_root(),
                export_path,
                itk,
                temeta=temeta
            ), miter
        )
    else:
        trainer = NekoAbstractModularJointEval(
            modcfg(
                root,
                find_data_root(),
                export_path,
                itk,
                maxT_overrider
            ), miter
        )
    if not auf:
        import torch
        trainer.modular_dict["pred"].model.UNK_SCR = torch.nn.Parameter(
            torch.ones_like(trainer.modular_dict["pred"].model.UNK_SCR) * -6000000)

    globalcache, mdict = trainer.pretest(0)
    return trainer, globalcache, mdict


# def launchtest(modcfg_dict, itr_override=None, miter=10000, rot=0, auf=True, maxT_overrider=None):
#     if (itr_override is not None):
#         itk = itr_override
#     trainer = NekoAbstractModularJointEval(modcfg_dict, miter)
#
#     if not auf:
#         trainer.modular_dict["pred"].model.UNK_SCR = torch.nn.Parameter(
#             torch.ones_like(trainer.modular_dict["pred"].model.UNK_SCR) * -6000000)
#
#     trainer.val(9, 9, rot)


def launchtest(argv, modcfg, itr_override=None, miter=10000, rot=0, auf=True, maxT_overrider=None):
    # from neko_2021_mjt.neko_abstract_jtr import neko_abstract_modular_joint_eval
    from neko_2021_mjt.neko_abstract_jtr import NekoAbstractModularJointEval
    from data_root import find_data_root;
    if (len(argv) > 2):
        export_path = argv[1];
        if (export_path == "None"):
            export_path = None;
        itk = argv[2];
        root = argv[3];
    else:
        export_path = find_export_root();
        itk = "latest";
        root = "jtrmodels";
    if (itr_override is not None):
        itk = itr_override;

    modscc = modcfg(
        root,
        find_data_root(),
        export_path,
        itk,
    )

    import yaml
    name = root.split('/')[7]
    new_name_dict = {
        "DUAL_a_Odancukmk7hdtfnp_r45_C_trinorm_dsa3": "OSTR_C2J_DTAOnly",
        "DUAL_a_Odancukmk7hnp_r45_C_trinorm_dsa3": "OSTR_C2J_BaseModel",
        "DUAL_a_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3": "OSTR_C2J_Full",
        "DUAL_a_Odancukmk8ahdtfnp_r45pttpt_C_trinorm_dsa3": "OSTR_C2J_FullLarge",
        "DUAL_b_Odancukmk8ahdtfnp_r45pttpt_C_trinorm_dsa3": "CSTR_FullLarge",
        "DUAL_ch_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3": "ZSCR_CTW_Full",
        "DUAL_chhw_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3": "ZSCR_Handwritten_Full"
    }
    new_name = new_name_dict[name]

    yaml.dump(modcfg(
        root,
        find_data_root(),
        export_path,
        itk,
    ), open(f"../../exp/{new_name}.yaml", 'w+'))

    exit(0)

    if (maxT_overrider is None):
        trainer = neko_abstract_modular_joint_eval(
            modcfg(
                root,
                find_data_root(),
                export_path,
                itk,
            ), miter
        );
    else:
        trainer = neko_abstract_modular_joint_eval(
            modcfg(
                root,
                find_data_root(),
                export_path,
                itk,
                maxT_overrider
            ), miter
        );
    if not auf:
        import torch
        trainer.modular_dict["pred"].model.UNK_SCR = torch.nn.Parameter(
            torch.ones_like(trainer.modular_dict["pred"].model.UNK_SCR) * -6000000)
    trainer.val(9, 9, rot);





def launchtest_image(src_path, export_path, itk, root, tskcfg, miter=10000, rot=0, auf=True):
    from neko_2021_mjt.eval_tasks.dan_eval_tasks import NekoOdanEvalTasks
    shutil.rmtree(export_path, ignore_errors=True)
    os.makedirs(export_path)
    tsk = NekoOdanEvalTasks(root, itk, None, tskcfg, miter)
    if not auf:
        import torch
        tsk.modular_dict["pred"].model.UNK_SCR = torch.nn.Parameter(
            torch.ones_like(tsk.modular_dict["pred"].model.UNK_SCR) * -6000000)
    proto, plabel, tdict, handle = tsk.get_proto_and_handle(0)

    images = glob.glob(os.path.join(src_path, "*.jpg"))
    bnames = [os.path.basename(i) for i in images]
    for i in range(len(images)):
        image_path = images[i]
        gt_path = os.path.join(export_path, bnames[i].replace("jpg", "txt"))
        text, _, beams = handle.test_image(image_path, proto, plabel, tdict, h=32, w=100)
        with open(gt_path, "w+") as fp:
            fp.writelines([text[0], str(beams)])
