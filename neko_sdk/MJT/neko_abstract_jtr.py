import os
import random
import time

import numpy
import torch
from torch.nn.parallel import parallel_apply

from neko_sdk.MJT.common import Updata_Parameters
from neko_sdk.MJT.neko_module_set import neko_module_set
from neko_sdk.MJT.utils import update
from neko_sdk.thirdparty.mmdetapply import multi_apply


class NekoAbstractModularJointTraining(neko_module_set):

    def set_routines(self, routine_cfgs):
        self.routines = []
        self.routine_names = []
        for rcfg in routine_cfgs:
            self.routine_names.append(rcfg)
            self.routines.append(routine_cfgs[rcfg]["routine"](routine_cfgs[rcfg]))

    def set_val_tasks(self, val_cfgs):
        self.val_tasks = []
        for vk in val_cfgs:
            self.val_tasks.append(val_cfgs[vk]["type"](None, None, self.modular_dict, val_cfgs[vk], 1000))

    def set_dataloader(self, datacfg, vitr):
        self.joint_dataloader = datacfg["loadertype"](datacfg, vitr)

    def setup(self, cfgs):
        root, self.val_each, self.vitr, self.vepoch = \
            cfgs["root"], cfgs["val_each"], cfgs["vitr"], cfgs["vepoch"]
        # set to "latest" for resuming, whatever does not make sense to start fresh.
        self.set_dataloader(cfgs["dataloader_cfg"], vitr=cfgs["vitr"])
        self.arm_modules(root, cfgs["modules"], cfgs["iterkey"])
        self.set_routines(cfgs["routine_cfgs"])
        self.set_val_tasks(cfgs["tasks"])

    def __init__(self,
                 cfgs):
        seed = 9
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)
        print("We are running from commit,", os.popen('git rev-parse HEAD').read())
        self.setup(cfgs)
        pass

        # ---------------------------------

    def val(self, nEpoch, batch_idx, vdbg=None):
        self.eval_mode()
        # torch.cuda.empty_cache()
        for vt in self.val_tasks:
            print(nEpoch, batch_idx)
            torch.cuda.empty_cache()
            with torch.no_grad():
                vt.test(vdbg=vdbg)
        torch.cuda.empty_cache()
        self.train_mode()

    def launch(self, rot, sample_batched, nEpoch, batch_idx):
        rot.fpbp(sample_batched, self.modular_dict, nEpoch, batch_idx)
        return []

    def tr_iter_amp(self, nEpoch, batch_idx, sample_batched):
        # torch.autograd.set_detect_anomaly(True)
        # data prepare
        zg_start = time.time()
        for modk in self.modular_dict:
            if (self.modular_dict[modk] == "NEP_skipped_NEP"):
                continue
            self.modular_dict[modk].zero_grad()

        routine_start = time.time()
        # multi_apply(self.launch,self.routines, sample_batched=sample_batched, nEpoch=nEpoch,
        #             batch_idx=batch_idx)
        #
        for routine in self.routines:
            routine.fpbp_amp(sample_batched, self.modular_dict, nEpoch, batch_idx)
        # reqnorm=[]
        # for modk in self.modular_dict:
        #     if(self.modular_dict[modk].save_each>0):
        #         reqnorm.append(self.modular_dict[modk])
        # multi_apply(normgrad,reqnorm)

        for modk in self.modular_dict:
            if (self.modular_dict[modk] == "NEP_skipped_NEP"):
                continue
            if (self.modular_dict[modk].save_each > 0):
                ng_start_ = time.time()
                self.modular_dict[modk].normgrad()
                # print(modk,time.time()-ng_start_)
        pu_start = time.time()
        try:
            Updata_Parameters(self.optimizers, frozen=[])
        except:
            print("Oops")
            exit(9)
        all_done = time.time()

        if (batch_idx % 100 == 9):
            print("[Timings]: zg:", routine_start - zg_start, "routines:", pu_start - routine_start, "pu:",
                  all_done - pu_start)

        for modk in self.modular_dict:
            if (self.modular_dict[modk] == "NEP_skipped_NEP"):
                continue
            self.modular_dict[modk].save_if_needed(nEpoch, batch_idx)

    def tr_iter(self, nEpoch, batch_idx, sample_batched):
        # torch.autograd.set_detect_anomaly(True)
        # data prepare
        zg_start = time.time()
        for modk in self.modular_dict:
            if (self.modular_dict[modk] == "NEP_skipped_NEP"):
                continue
            self.modular_dict[modk].zero_grad()

        routine_start = time.time()
        # multi_apply(self.launch,self.routines, sample_batched=sample_batched, nEpoch=nEpoch,
        #             batch_idx=batch_idx)
        #
        for routine in self.routines:
            routine.fpbp(sample_batched, self.modular_dict, nEpoch, batch_idx)
        # reqnorm=[]
        # for modk in self.modular_dict:
        #     if(self.modular_dict[modk].save_each>0):
        #         reqnorm.append(self.modular_dict[modk])
        # multi_apply(normgrad,reqnorm)

        for modk in self.modular_dict:
            if (self.modular_dict[modk] == "NEP_skipped_NEP"):
                continue
            if (self.modular_dict[modk].save_each > 0):
                ng_start_ = time.time()
                self.modular_dict[modk].normgrad()
                # print(modk,time.time()-ng_start_)
        pu_start = time.time()
        try:
            Updata_Parameters(self.optimizers, frozen=[])
        except:
            print("Oops")
            exit(9)
        all_done = time.time()

        if (batch_idx % 100 == 9):
            print("[Timings]: zg:", routine_start - zg_start, "routines:", pu_start - routine_start, "pu:",
                  all_done - pu_start)

        for modk in self.modular_dict:
            if (self.modular_dict[modk] == "NEP_skipped_NEP"):
                continue
            self.modular_dict[modk].save_if_needed(nEpoch, batch_idx)

    def train(self, dbgpath, vdbg=None, flag=None):
        torch.backends.cudnn.benchmark = True

        for modk in self.modular_dict:
            if (self.modular_dict[modk] == "NEP_skipped_NEP"):
                continue
            self.modular_dict[modk].cuda()
            self.modular_dict[modk].train()
        for nEpoch in range(0, self.vepoch):
            for batch_idx in range(self.vitr):
                if (flag is None or flag == False):
                    flag = (batch_idx > 0) or (dbgpath is not None)
                if (flag and batch_idx % self.val_each == 0):
                    self.val(nEpoch, batch_idx, vdbg=vdbg)
                data_start = time.time()
                sample_batched = self.joint_dataloader.next()
                data_time = time.time() - data_start
                itr_start = time.time()
                if (dbgpath is not None):
                    sample_batched["debug_path"] = dbgpath
                if (vdbg is not None):
                    sample_batched["vdbg"] = vdbg

                # for d in sample_batched:
                #     if(type(sample_batched[d])==torch.tensor):
                #         sample_batched[d]=sample_batched[d].cuda()
                self.tr_iter(nEpoch, batch_idx, sample_batched)
                itr_time = time.time() - itr_start

                # print(torch.backends.cudnn.benchmark)
                if (batch_idx % 100 == 9):
                    print("datatime", data_time, "itrtime", itr_time, "all", time.time() - data_start)
            Updata_Parameters(self.optimizer_schedulers, frozen=[])
            self.val(nEpoch, "Final")

            # torch.backends.cudnn.benchmark = False
            for modk in self.modular_dict:
                if (self.modular_dict[modk] == "NEP_skipped_NEP"):
                    continue
                self.modular_dict[modk].save(nEpoch)


class NekoAbstractModularJointEval(neko_module_set):

    def set_val_tasks(self, val_cfgs, mitr):
        self.val_tasks = []
        self.val_keys = []
        for vk in val_cfgs:
            self.val_keys.append(vk)
            self.val_tasks.append(val_cfgs[vk]["type"](None, None, self.modular_dict, val_cfgs[vk], mitr))

    def test_img(self, id, image_path, globalcache, h=32, w=100):
        return self.val_tasks[id].test_image(image_path, globalcache)

    def test_img_top_k(self, id, image_path, attover_paths, globalcache, h=32, w=100):
        return self.val_tasks[id].test_top_k(image_path, attover_paths, globalcache)

    def pretest(self, id):
        self.eval_mode()
        return self.val_tasks[id].testready()

    def __init__(self,
                 cfgs, mitr):
        root = \
            cfgs["root"]
        # set to "latest" for resuming, whatever does not make sense to start fresh.
        self.arm_modules(root, cfgs["modules"], cfgs["iterkey"])
        for mk in self.modular_dict:
            if (self.modular_dict[mk] == "NEP_skipped_NEP"):
                continue
            self.modular_dict[mk].model.cuda()
        if ("export_path" in cfgs and cfgs["export_path"] is not None):
            for k in cfgs["tasks"]:
                cfgs["tasks"][k]["export_path"] = cfgs["export_path"]
        self.set_val_tasks(cfgs["tasks"], mitr)
        pass

        # ---------------------------------

    def val(self, nEpoch, batch_idx, rot=0):
        self.eval_mode()
        tasklogs = {}
        for vid in range(len(self.val_tasks)):
            print(self.val_keys[vid], nEpoch, batch_idx, "Starts", "------------------------")
            torch.cuda.empty_cache()
            with torch.no_grad():
                tasklogs[vid] = self.val_tasks[vid].test(rot, logname="E" + str(nEpoch) + "_I" + str(batch_idx))
            print("------------------------------------------------------")

        self.train_mode()

    def vis(self, nEpoch, batch_idx, rot=0):
        self.eval_mode()
        for vt in self.val_tasks:
            print(nEpoch, batch_idx)
            torch.cuda.empty_cache()
            vt.visualize(rot)
        self.train_mode()

        # ---------------------------------

    def valt(self, nEpoch, batch_idx):
        self.train_mode()
        for vt in self.val_tasks:
            print(nEpoch, batch_idx)
            with torch.no_grad():
                torch.cuda.empty_cache()
                vt.test()
        self.train_mode()


# some routines may change shared module state.
class NekoModularJointTrainingSemipara(NekoAbstractModularJointTraining):
    def tr_iter(self, nEpoch, batch_idx, sample_batched):
        # torch.autograd.set_detect_anomaly(True)
        # data prepare
        # torch.backends.cudnn.benchmark=True

        zg_start = time.time()

        for modk in self.modular_dict:
            if (self.modular_dict[modk] == "NEP_skipped_NEP"):
                continue
            self.modular_dict[modk].zero_grad()
        routine_start = time.time()

        # multi_apply(self.launch,self.routines, sample_batched=sample_batched, nEpoch=nEpoch,
        #             batch_idx=batch_idx)

        # i=0
        for routine in self.routines:
            # rs = time.time()
            routine.fpbp(sample_batched, self.modular_dict, nEpoch, batch_idx)
        #
        #     # print(self.routine_names[i],time.time()-rs)
        #     # i += 1
        ng_start = time.time()

        for modk in self.modular_dict:
            if (self.modular_dict[modk] == "NEP_skipped_NEP"):
                continue
            if (self.modular_dict[modk].save_each > 0):
                self.modular_dict[modk].normgrad()
        pu_start = time.time()
        multi_apply(update, self.optimizers)
        # try:
        #     Updata_Parametersd(self.optimizers,self.optnames, frozen=[])
        # except:
        #     print("Oops")
        #     exit(9)
        all_done = time.time()
        if (batch_idx % 100 == 9):
            print("[Timings]: zg:", routine_start - zg_start, "routines:", pu_start - routine_start, "pu:",
                  all_done - pu_start)

        for modk in self.modular_dict:
            if (self.modular_dict[modk] == "NEP_skipped_NEP"):
                continue
            self.modular_dict[modk].save_if_needed(nEpoch, batch_idx)


class NekoModularJointTrainingPara(NekoAbstractModularJointTraining):
    def tr_iter(self, nEpoch, batch_idx, sample_batched):
        # torch.autograd.set_detect_anomaly(True)
        # data prepare
        # torch.backends.cudnn.benchmark=True

        zg_start = time.time()

        for modk in self.modular_dict:
            if (self.modular_dict[modk] == "NEP_skipped_NEP"):
                continue
            self.modular_dict[modk].zero_grad()
        routine_start = time.time()

        multi_apply(self.launch, self.routines, sample_batched=sample_batched, nEpoch=nEpoch,
                    batch_idx=batch_idx)

        # i=0
        # for routine in self.routines:
        #     # rs = time.time()
        #     routine.fpbp(sample_batched,self.modular_dict,nEpoch,batch_idx)
        #
        #     # print(self.routine_names[i],time.time()-rs)
        #     # i += 1
        ng_start = time.time()

        # for modk in self.modular_dict:
        #     if(self.modular_dict[modk].save_each>0):
        #         self.modular_dict[modk].normgrad()
        pu_start = time.time()
        multi_apply(update, self.optimizers)
        # try:
        #     Updata_Parameters(self.optimizers, frozen=[])
        # except:
        #     print("Oops")
        #     exit(9)
        all_done = time.time()
        if (batch_idx % 100 == 9):
            print("[Timings]: zg:", routine_start - zg_start, "routines:", pu_start - routine_start, "pu:",
                  all_done - pu_start)

        for modk in self.modular_dict:
            if (self.modular_dict[modk] == "NEP_skipped_NEP"):
                continue
            self.modular_dict[modk].save_if_needed(nEpoch, batch_idx)


class NekoModularJointTrainingPara2(NekoAbstractModularJointTraining):
    def launch(self, rot, sample_batched, nEpoch, batch_idx):
        l = rot.fp(sample_batched, self.modular_dict, nEpoch, batch_idx, "cuda")
        return [l]

    def tr_iter(self, nEpoch, batch_idx, sample_batched):
        # torch.autograd.set_detect_anomaly(True)
        # data prepare
        # torch.backends.cudnn.benchmark=True

        zg_start = time.time()

        for modk in self.modular_dict:
            self.modular_dict[modk].zero_grad()
        routine_start = time.time()

        losses = multi_apply(self.launch, self.routines, sample_batched=sample_batched, nEpoch=nEpoch,
                             batch_idx=batch_idx)
        loss = torch.stack([loss[0] for loss in losses]).sum()

        #
        # loss=0
        # for routine in self.routines:
        #     # rs = time.time()
        #     loss+=routine.fp(sample_batched,self.modular_dict,nEpoch,batch_idx)
        #     # print(self.routine_names[i],time.time()-rs)
        #     # i += 1
        #
        loss.backward()
        ng_start = time.time()

        for modk in self.modular_dict:
            if (self.modular_dict[modk].save_each > 0):
                self.modular_dict[modk].normgrad()
        pu_start = time.time()
        # multi_apply(update,self.optimizers)
        try:
            Updata_Parameters(self.optimizers, frozen=[])
        except:
            print("Oops")
        #     exit(9)
        all_done = time.time()
        if (batch_idx % 100 == 9):
            print("[Timings]: zg:", routine_start - zg_start, "routines:", pu_start - routine_start, "pu:",
                  all_done - pu_start)

        for modk in self.modular_dict:
            self.modular_dict[modk].save_if_needed(nEpoch, batch_idx)


class NekoModularJointTrainingPara3(NekoAbstractModularJointTraining):
    def launch(self, rot, sample_batched, nEpoch, batch_idx):
        l = rot.fp(sample_batched, self.modular_dict, nEpoch, batch_idx, "cuda")
        return [l]

    def tr_iter(self, nEpoch, batch_idx, sample_batched):
        # torch.autograd.set_detect_anomaly(True)
        # data prepare
        # torch.backends.cudnn.benchmark=True

        zg_start = time.time()

        for modk in self.modular_dict:
            self.modular_dict[modk].zero_grad()
        routine_start = time.time()
        inp = [[sample_batched, self.modular_dict, nEpoch, batch_idx] for _ in self.routines]
        dev = ["cuda" for _ in self.routines]
        parallel_apply(self.routines, inp, devices=dev)

        #
        # loss=0
        # for routine in self.routines:
        #     # rs = time.time()
        #     loss+=routine.fp(sample_batched,self.modular_dict,nEpoch,batch_idx)
        #     # print(self.routine_names[i],time.time()-rs)
        #     # i += 1
        #
        # loss.backward()
        ng_start = time.time()

        for modk in self.modular_dict:
            if (self.modular_dict[modk].save_each > 0):
                self.modular_dict[modk].normgrad()
        pu_start = time.time()
        # multi_apply(update,self.optimizers)
        try:
            Updata_Parameters(self.optimizers, frozen=[])
        except:
            print("Oops")
        #     exit(9)
        all_done = time.time()
        if (batch_idx % 100 == 9):
            print("[Timings]: zg:", routine_start - zg_start, "routines:", pu_start - routine_start, "pu:",
                  all_done - pu_start)

        for modk in self.modular_dict:
            if (self.modular_dict[modk] == "NEP_skipped_NEP"):
                continue
            self.modular_dict[modk].save_if_needed(nEpoch, batch_idx)
