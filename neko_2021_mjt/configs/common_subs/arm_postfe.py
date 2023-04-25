from neko_2021_mjt.configs.modules.config_cls_emb_loss import config_cls_emb_loss2, config_cls_emb_lossohem
from neko_2021_mjt.configs.modules.config_dtd_xos_mk5 import config_dtdmk5
from neko_2021_mjt.configs.modules.config_ocr_sampler import config_ocr_sampler
from neko_2021_mjt.configs.modules.config_ospred import config_linxos
from neko_2021_mjt.configs.modules.config_prototyper import config_prototyper


def arm_rest_commonwops(srcdst, prefix, wemb=0.3):
    srcdst["DTD"] = config_dtdmk5()
    srcdst["pred"] = config_linxos()
    srcdst["loss_cls_emb"] = config_cls_emb_loss2(wemb)
    return srcdst


def arm_rest_commonwopsd(srcdst, prefix, wemb=0):
    srcdst["DTD"] = config_dtdmk5()
    srcdst["pred"] = config_linxos()
    srcdst["loss_cls_emb"] = config_cls_emb_loss2(wemb)
    return srcdst


def arm_rest_commonwops_ohem(srcdst, prefix):
    srcdst["DTD"] = config_dtdmk5()
    srcdst["pred"] = config_linxos()
    srcdst["loss_cls_emb"] = config_cls_emb_lossohem()
    return srcdst


def arm_rest_commonwop(srcdst, prefix, maxT, capacity, tr_meta_path, wemb=0.3):
    srcdst = arm_rest_commonwops(srcdst, prefix, wemb)
    srcdst["Latin_62_sampler"] = config_ocr_sampler(tr_meta_path, capacity)
    return srcdst


def arm_rest_common(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path, wemb=0.3):
    srcdst = arm_rest_commonwop(srcdst, prefix, maxT, capacity, tr_meta_path, wemb=wemb)
    srcdst["prototyper"] = config_prototyper(capacity, feat_ch)
    return srcdst
