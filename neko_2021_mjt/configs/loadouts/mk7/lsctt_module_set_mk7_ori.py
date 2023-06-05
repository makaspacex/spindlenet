from neko_2021_mjt.configs.bogo_modules.config_res_binorm import config_bogo_resbinorm
from neko_2021_mjt.configs.common_subs.arm_post_fe_shared_prototyper import arm_shared_prototyper_np
from neko_2021_mjt.configs.common_subs.arm_postfe import arm_rest_commonwop
from neko_2021_mjt.configs.modules.config_cam_stop import config_cam_stop
from neko_2021_mjt.configs.modules.config_fe_db import config_fe_r45_binorm_orig
from neko_2021_mjt.configs.modules.config_sa import config_sa_mk3


def arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7(prefix, maxT, capacity, feat_ch, tr_meta_path, srcdst=None, expf=1,
                                                  fecnt=3, wemb=0.3):

    srcdst = {} if srcdst is None else srcdst

    srcdst[prefix + "feature_extractor_container"] = config_fe_r45_binorm_orig(3, feat_ch, cnt=fecnt)
    srcdst[prefix + "feature_extractor_cco"] = config_bogo_resbinorm(prefix + "feature_extractor_container", "res1")
    srcdst[prefix + "feature_extractor_proto"] = config_bogo_resbinorm(prefix + "feature_extractor_container", "res2")
    srcdst[prefix + "GA"] = config_sa_mk3(feat_ch=32)
    srcdst[prefix + "TA"] = config_cam_stop(maxT, feat_ch=feat_ch, scales=[
        [int(32), 16, 64],
        [int(128), 8, 32],
        [int(feat_ch), 8, 32]
    ])
    srcdst = arm_rest_commonwop(srcdst, prefix, maxT, capacity, tr_meta_path, wemb=wemb)
    srcdst = arm_shared_prototyper_np(
        srcdst, prefix, capacity, feat_ch,
        prefix + "feature_extractor_proto",
        prefix + "GA",
        use_sp=False
    )
    return srcdst