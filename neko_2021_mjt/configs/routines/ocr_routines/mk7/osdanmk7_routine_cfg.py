from neko_2021_mjt.routines.ocr_routines.mk7.osdan_routine_mk7 import Nekohdos2croutinecfmk7, \
    NekoHdos2cEvalRoutineCfmk7, Nekohdos2cRoutineCfmk7dtf


def osdanmk7_ocr_routine(sampler_name, prototyper_name, feature_extractor_name, seq_name,
                         CAMname, pred_name, loss_name, label_name, image_name, log_path, log_each, name, maxT):
    return \
        {

            "maxT": maxT,
            "name": name,
            "routine": Nekohdos2croutinecfmk7,
            "mod_cvt_dicts":
                {
                    "sampler": sampler_name,
                    "prototyper": prototyper_name,
                    "feature_extractor": feature_extractor_name,
                    "CAM": CAMname,
                    "seq": seq_name,
                    "preds": pred_name,
                    "losses": loss_name,
                },
            "inp_cvt_dicts":
                {
                    "label": label_name,
                    "image": image_name,
                },
            "log_path": log_path,
            "log_each": log_each,
        }


def osdanmk7dt_ocr_routine(sampler_name, prototyper_name, feature_extractor_name, seq_name,
                           CAMname, pred_name, loss_name, label_name, image_name, log_path, log_each, name, maxT):
    return \
        {

            "maxT": maxT,
            "name": name,
            "routine": Nekohdos2cRoutineCfmk7dtf,
            "mod_cvt_dicts":
                {
                    "sampler": sampler_name,
                    "prototyper": prototyper_name,
                    "feature_extractor": feature_extractor_name,
                    "CAM": CAMname,
                    "seq": seq_name,
                    "preds": pred_name,
                    "losses": loss_name,
                },
            "inp_cvt_dicts":
                {
                    "label": label_name,
                    "image": image_name,
                },
            "log_path": log_path,
            "log_each": log_each,
        }


def osdanmk7_eval_routine_cfg(sampler_name, prototyper_name, feature_extractor_name,
                              CAMname, seq_name, pred_name, loss_name, label_name, image_name, log_path, name, maxT,
                              measure_rej=False):
    return \
        {
            "name": name,
            "maxT": maxT,
            "routine": NekoHdos2cEvalRoutineCfmk7,
            "mod_cvt_dicts":
                {
                    "sampler": sampler_name,
                    "prototyper": prototyper_name,
                    "feature_extractor": feature_extractor_name,
                    "CAM": CAMname,
                    "seq": seq_name,
                    "preds": pred_name,
                    "losses": loss_name,
                },
            "inp_cvt_dicts":
                {
                    "label": label_name,
                    "image": image_name,
                    "proto": "proto",
                    #            "semb": "semb",
                    "plabel": "plabel",
                    "tdict": "tdict",
                },
            "measure_rej": measure_rej,
            "log_path": log_path,
        }
