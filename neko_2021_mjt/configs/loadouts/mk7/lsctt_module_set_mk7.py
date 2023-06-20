from neko_2021_mjt.configs.bogo_modules.config_res_binorm import config_bogo_resbinorm
from neko_2021_mjt.configs.common_subs.arm_post_fe_shared_prototyper import arm_shared_prototyper_np
from neko_2021_mjt.configs.common_subs.arm_postfe import arm_rest_commonwop
from neko_2021_mjt.configs.modules.config_cam_stop import config_cam_stop
from neko_2021_mjt.configs.modules.config_fe_db import config_fe_r45_binorm_orig
from neko_2021_mjt.configs.modules.config_sa import config_sa_mk3


def arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7(prefix, maxT, capacity, feat_ch, tr_meta_path, srcdst=None, expf=1,
                                                  fecnt=3, wemb=0.3):
    srcdst = {} if srcdst is None else srcdst

    srcdst["feature_extractor_container"] = config_fe_r45_binorm_orig(3, feat_ch, cnt=fecnt)
    srcdst["feature_extractor_cco"] = config_bogo_resbinorm("feature_extractor_container", "res1")
    srcdst["feature_extractor_proto"] = config_bogo_resbinorm("feature_extractor_container", "res2")
    srcdst["GA"] = config_sa_mk3(feat_ch=32)
    srcdst["TA"] = config_cam_stop(maxT, feat_ch=feat_ch, scales=[
        [int(32), 16, 64],
        [int(128), 8, 32],
        [int(feat_ch), 8, 32]
    ])
    srcdst = arm_rest_commonwop(srcdst, prefix, maxT, capacity, tr_meta_path, wemb=wemb)
    srcdst = arm_shared_prototyper_np(
        srcdst, prefix, capacity, feat_ch,
        "feature_extractor_proto",
        "GA",
        use_sp=False
    )
    return srcdst


def arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7_wide_64(prefix, maxT, capacity, feat_ch, tr_meta_path, srcdst=None, expf=1, fecnt=3, wemb=0.3, cam_ch=64):
    
    srcdst = {} if srcdst is None else srcdst
    # ori
    # ch_overid_num = [32,32, 64,128,256,512]
    ch_overid_num = [32, 64, 64,128,256,feat_ch]
    
    srcdst["feature_extractor_container"] = config_fe_r45_binorm_orig(3, feat_ch, cnt=fecnt,ch_overid_num=ch_overid_num)
    srcdst["feature_extractor_cco"] = config_bogo_resbinorm("feature_extractor_container", "res1")
    srcdst["feature_extractor_proto"] = config_bogo_resbinorm("feature_extractor_container", "res2")
    srcdst["GA"] = config_sa_mk3(feat_ch=ch_overid_num[1])
    srcdst["TA"] = config_cam_stop(maxT, feat_ch=feat_ch, scales=[
        [int(ch_overid_num[1]), 16, 64],
        [int(ch_overid_num[3]), 8, 32],
        [int(feat_ch), 8, 32]
    ],cam_ch=cam_ch)
    srcdst = arm_rest_commonwop(srcdst, prefix, maxT, capacity, tr_meta_path, wemb=wemb)
    srcdst = arm_shared_prototyper_np(
        srcdst, prefix, capacity, feat_ch,
        "feature_extractor_proto",
        "GA",
        use_sp=False
    )
    return srcdst
