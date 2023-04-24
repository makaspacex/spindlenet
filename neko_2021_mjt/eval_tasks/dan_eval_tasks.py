import os.path
import time

import cv2
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from neko_2020nocr.result_renderer import render_words
from neko_2021_mjt.neko_abstract_jtr import NekoModuleSet
from neko_2021_mjt.neko_laser import NekoLaser
from neko_sdk.ocr_modules.charset.chs_cset import t1_3755
from neko_sdk.ocr_modules.charset.etc_cset import latin62
from neko_sdk.ocr_modules.img_eval import keepratio_resize
from neko_sdk.ocr_modules.neko_prototyper_gen2.neko_label_sampler import neko_prototype_sampler_static


class NekoAbstractEvalTasks(NekoModuleSet):
    def setupthis(self, cfgs):
        pass

    def test_ds(self, test_loader, dsname, miter=1000, debug=False, dbgpath=None, rot=0):
        print("wrong path")
        pass

    def vis_ds(self, test_loader, dsname, miter=1000, debug=False, dbgpath=None, rot=0):
        print("wrong path")
        pass

    def test(self, rot=0, vdbg=None):
        for dsname in self.datasets["datasets"]:
            print(dsname, "starts")
            cfg = self.datasets["datasets"][dsname]
            train_data_set = cfg['type'](**cfg['ds_args'])
            cfg['dl_args']['num_workers'] = 0
            train_loader = DataLoader(train_data_set, **cfg['dl_args'])
            self.test_ds(train_loader, dsname, self.miter, rot=rot, debug=vdbg)
            print(dsname, "ends")

    def visualize(self, rot=0, vdbg=None):
        for dsname in self.datasets["datasets"]:
            print(dsname, "starts")
            cfg = self.datasets["datasets"][dsname]
            train_data_set = cfg['type'](**cfg['ds_args'])
            train_loader = DataLoader(train_data_set, **cfg['dl_args'])
            self.vis_ds(train_loader, dsname, self.miter, rot=rot, debug=vdbg)
            print(dsname, "ends")

    def __init__(self, root, itrkey, modulars, cfgs, miter):
        self.setupthis(cfgs)
        self.eval_routine = cfgs["routine_cfgs"]["routine"](cfgs["routine_cfgs"])
        self.datasets = cfgs["datasets"]
        try:
            self.export_path = cfgs["export_path"]
        except:
            self.export_path = None

        self.miter = miter
        if (modulars is None):
            self.arm_modules(root, cfgs["modules"], itrkey)
        else:
            self.modulars = modulars
        pass


class NekoOdanEvalTasks(NekoAbstractEvalTasks):
    def setupthis(self, cfgs):
        self.temeta_args = cfgs["temeta"]
        self.protoname = cfgs["protoname"]

    def exportvis(self, data, label, gmaps, results, all, export_path, mdict):
        texts, etc, beams = results

        for i in range(len(data)):
            name = os.path.join(export_path, str(all + i))
            im = (data[i].cpu().detach() * 255).permute(1, 2, 0).numpy().astype(np.uint8)
            im, ned = render_words(mdict, set(latin62.union(t1_3755)), im, label[i], beams[i])
            if (len(gmaps) == 0):
                continue
            gm = (torch.cat(gmaps[i], 1) * 255).permute(1, 2, 0).numpy().astype(np.uint8)

            cv2.imwrite(name + ".jpg", im)
            cv2.imwrite(name + "gm" + ".png", gm)

            if ("xtra_ims" in etc):
                for n in etc["xtra_ims"][i]:
                    cv2.imwrite(name + n + ".jpg", etc["xtra_ims"][i][n])

            with open(name + ".txt", "w+") as fp:
                fp.write(label[i] + "\n")
                fp.write(texts[i] + "\n")
                # fp.write(str(probs[i]) + "\n")
                fp.write(str(beams[i]) + "\n")

        pass

    def export(self, data, label, results, all, export_path, mdict):
        texts, etc, beams = results

        for i in range(len(data)):
            name = os.path.join(export_path, str(all + i))
            im = (data[i].cpu().detach() * 255).permute(1, 2, 0).numpy().astype(np.uint8)
            im, ned = render_words(mdict, set(latin62.union(t1_3755)), im, label[i], beams[i])
            cv2.imwrite(name + ".jpg", im)
            if ("xtra_ims" in etc):
                for n in etc["xtra_ims"][i]:
                    cv2.imwrite(name + n + ".png", etc["xtra_ims"][i][n])

            with open(name + ".txt", "w+") as fp:
                fp.write(label[i] + "\n")
                fp.write(texts[i] + "\n")
                # fp.write(str(probs[i]) + "\n")
                fp.write(str(beams[i]) + "\n")

        pass

    def get_proto_and_handle(self, rot):
        sampler = neko_prototype_sampler_static(self.temeta_args, None)
        normims, plabel, tdict = sampler.dump_all()
        proto = self.modulars[self.protoname](normims, rot)
        return proto, plabel, tdict, self

    # A primitive interface for for single image evaluation and                    answer "What if" questions
    def test_image(self, image_path, globalcache, h=32, w=128):
        im = cv2.imread(image_path)
        im, bmask = keepratio_resize(im, h, w, rgb=True)
        tim = torch.tensor(im).permute(2, 0, 1).unsqueeze(0) / 255.
        texts, probs, beams = self.eval_routine.test(
            input_dict={"image": tim.cuda(), "bmask": bmask, "label": None, **globalcache},
            modular_dict=self.modulars
        )
        return texts[0]

    def vis_ds(self, test_loader, dsname, miter=1000, debug=None, dbgpath=None, rot=0):
        tmetastart = time.time()
        with torch.no_grad():
            global_cache = self.eval_routine.pretest(self.modulars, metaargs=self.temeta_args, rot=rot)
        visualizer = NekoLaser(self.eval_routine, self.modulars)
        mdict = torch.load(self.temeta_args["meta_path"])

        # doing some irrelevant shit.
        if (global_cache is None):
            global_cache = {}

        tmetaend = time.time()
        self.eval_routine.clear_loggers()

        fwdstart = time.time()
        idi = 0
        all = 0
        for sample_batched in test_loader:
            if idi > miter:
                break
            idi += 1
            texts, etc, beams = self.eval_routine.test(input_dict={**sample_batched, **global_cache
                                                                   }, modular_dict=self.modulars, vdbg=debug,
                                                       )
            gmaps = visualizer.vis_chars(input_dict={**sample_batched, **global_cache}, modular_dict=self.modulars)

            if (self.export_path is not None):
                export_path = os.path.join(self.export_path, dsname)
                os.makedirs(export_path, exist_ok=True)
                self.exportvis(sample_batched["image"], sample_batched["label"], gmaps, [texts, etc, beams], all,
                               export_path, mdict)
            if ("label" in sample_batched):
                all += len(sample_batched["label"])
            else:
                all += len(sample_batched["labels"])

        fwdend = time.time()
        print((fwdend - fwdstart) / all, all)
        return self.eval_routine.ret_log()

    def testready(self):
        global_cache = self.eval_routine.pretest(self.modulars, metaargs=self.temeta_args, rot=False)
        mdict = torch.load(self.temeta_args["meta_path"])
        return global_cache, mdict

    def test_ds(self, test_loader, dsname, miter=1000, debug=None, dbgpath=None, rot=0):

        tmetastart = time.time()
        global_cache, mdict = self.testready()

        # doing some irrelevant shit.
        if (global_cache is None):
            global_cache = {}

        tmetaend = time.time()
        self.eval_routine.clear_loggers()

        fwdstart = time.time()
        idi = 0
        all = 0
        for sample_batched in test_loader:
            if idi > miter:
                break
            idi += 1
            texts, etc, beams = self.eval_routine.test(input_dict={**sample_batched, **global_cache
                                                                   }, modular_dict=self.modulars, vdbg=debug,
                                                       )
            if (self.export_path is not None):
                export_path = os.path.join(self.export_path, dsname)
                os.makedirs(export_path, exist_ok=True)
                self.export(sample_batched["image"], sample_batched["label"], [texts, etc, beams], all, export_path,
                            mdict)
            if ("label" in sample_batched):
                all += len(sample_batched["label"])
            else:
                all += len(sample_batched["labels"])

        fwdend = time.time()
        print((fwdend - fwdstart) / all, all, "FPS:", 1 / ((fwdend - fwdstart) / all))
        print((fwdend - tmetastart) / all, all)

        return self.eval_routine.ret_log()


class NekoOdanEvalTasksMk8(NekoOdanEvalTasks):
    # I think this changed....
    def get_proto_and_handle(self, rot):
        sampler = neko_prototype_sampler_static(self.temeta_args, None)
        normims, plabel, tdict = sampler.dump_all(use_sp=False)
        proto = self.modulars[self.protoname](normims, rot)
        return proto, plabel, tdict, self
