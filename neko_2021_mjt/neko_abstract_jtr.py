import os
import time

import torch
from torch import nn
from torch.nn.parallel import parallel_apply

from neko_2020nocr.dan.common.common import Updata_Parameters
from neko_sdk.thirdparty.mmdetapply import multi_apply

import copy

class NekoModular(object):
    def __init__(self, path, name, module, save_each=20000, prefix=""):
        self.path = path
        self.model = module
        self.name = name
        self.save_each = save_each
        self.prefix = prefix
        
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))

    def get_torch_module_dict(self):
        if (isinstance(self.model, nn.Module)):
            return self.model
        else:
            return None

    def train(self, training=True):
        self.model.train(training)

    def eval(self):
        self.model.eval()

    def normgrad(self):
        if self.save_each > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), 20, 2)

    def cuda(self):
        self.model.cuda()

    def zero_grad(self):
        if self.save_each > 0:
            for param in self.model.parameters():
                param.grad = None

    def load(self, w_full_path):
        p = w_full_path
        try:
            self.model.load_state_dict(torch.load(p).state_dict())
        except Exception as e:
            print(self.name, "cannot load", "itr", p, f", starting fresh. {e}")

    def save(self, nEpoch):
        if (self.save_each > 0):
            torch.save(self.model, self.path + '_E{}.pth'.format(nEpoch))
            torch.save(self.model, self.path + 'latest.pth')

    def save_if_needed(self, nEpoch, batch_idx):
        if (self.save_each > 0 and batch_idx % self.save_each == 0):
            print("Saving", self.path + '_E{}_I{}.pth'.format(nEpoch, batch_idx))
            torch.save(self.model, self.path + '_E{}_I{}.pth'.format(nEpoch, batch_idx))
            torch.save(self.model, self.path + 'latest.pth')

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class NekoBogoModular:
    def __init__(self, forwardable):
        # you should never haver a
        self.model = forwardable
        self.save_each = -9

    def get_torch_module_dict(self):
        try:
            return self.model.get_torch_module_dict()
        except:
            return None

        pass

    def train(self, training=True):
        pass

    def eval(self):
        pass

    def normgrad(self):
        pass

    def zero_grad(self):
        pass

    def load(self, itrkey):
        pass

    def cuda(self):
        pass

    def save(self, nEpoch):
        pass

    def save_if_needed(self, nEpoch, batch_idx):
        pass

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class NekoModuleSet(object):
    def arm_modules(self, root, modcfgs, itrkey, prefix=""):
        self.optimizers = []
        self.optnames = []
        self.optimizer_schedulers = []
        self.modular_dict = {}
        self.bogo_modular_list = []
        for name in modcfgs:
            cfg = modcfgs[name]
            modp = os.path.join(root, name)
            if ("bogo_mod" in cfg):
                self.bogo_modular_list.append(name)
            else:
                mod, opt, opts = cfg["modular"](cfg["args"], modp, modp)
                self.modular_dict[name] = NekoModular(modp, name, mod, cfg["save_each"])
                w_full_path = os.path.join(root, f"{prefix}{name}{itrkey}.pth")
                self.modular_dict[name].load(w_full_path)
                if (opt is not None):
                    self.optimizers.append(opt)
                    self.optnames.append(name)
                    self.optimizer_schedulers.append(opts)
        # make sure we have collected real modules.
        for name in self.bogo_modular_list:
            try:
                cfg = modcfgs[name]
                # bogo modules are re-combination of parts of existing modules.
                mod = cfg["bogo_mod"](cfg["args"], self.modular_dict)
                self.modular_dict[name] = NekoBogoModular(mod)
            except Exception as e:
                print(e)

    def eval_mode(self):
        for modk in self.modular_dict:
            self.modular_dict[modk].eval()

    def train_mode(self):
        for modk in self.modular_dict:
            self.modular_dict[modk].train()


def update(opt):
    try:
        opt.step()
    except:
        print("Oops", opt)
        exit(9)
    return []


def normgrad(mod):
    mod.normgrad()
    return []


class NekoAbstractModularJointTraining(NekoModuleSet):
    def __init__(self, cfgs):
        self.setup(cfgs)

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
            self.modular_dict[modk].save_if_needed(nEpoch, batch_idx)

    def tr_iter(self, nEpoch, batch_idx, sample_batched):
        # torch.autograd.set_detect_anomaly(True)
        # data prepare
        zg_start = time.time()
        for modk in self.modular_dict:
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
            self.modular_dict[modk].save_if_needed(nEpoch, batch_idx)

    def train(self, dbgpath, vdbg=None, flag=None):
        torch.backends.cudnn.benchmark = True

        for modk in self.modular_dict:
            self.modular_dict[modk].cuda()
        for modk in self.modular_dict:
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
                
                print(f"Epoch:{nEpoch+1}/{self.vepoch} Iter:{batch_idx+1}/{self.vitr} datatime {data_time} itrtime{itr_time} all {time.time() - data_start}")
                
            Updata_Parameters(self.optimizer_schedulers, frozen=[])
            self.val(nEpoch, "Final")
            # torch.backends.cudnn.benchmark = False
            for modk in self.modular_dict:
                self.modular_dict[modk].save(nEpoch)


class NekoAbstractModularJointEval(NekoModuleSet):
    def __init__(self, cfgs, miter):

        self.ori_cfgs = copy.deepcopy(cfgs)

        root = cfgs["root"]
        # set to "latest" for resuming, whatever does not make sense to start fresh.
        if "prefix" not in cfgs:
            print(f"cfgs does not contain prefix key.")
            cfgs['prefix'] = ""
        
        self.arm_modules(root, cfgs["modules"], cfgs["iterkey"], prefix=cfgs['prefix'])
        for mk in self.modular_dict:
            self.modular_dict[mk].model.cuda()
        if "export_path" in cfgs and cfgs["export_path"] is not None:
            for k in cfgs["tasks"]:
                cfgs["tasks"][k]["export_path"] = cfgs["export_path"]

        self.set_val_tasks(cfgs["tasks"], miter)

    def set_val_tasks(self, val_cfgs, miter):
        self.val_tasks = []
        for vk in val_cfgs:
            self.val_tasks.append(val_cfgs[vk]["type"](None, None, self.modular_dict, val_cfgs[vk], miter))

    def test_img(self, id, image_path, globalcache, h=32, w=100):
        return self.val_tasks[id].test_image(image_path, globalcache)

    def pretest(self, id):
        self.eval_mode()
        with torch.no_grad():
            tester = self.val_tasks[id].testready()
        return tester

    def val(self, nEpoch, batch_idx, rot=0):
        self.eval_mode()
        for vt in self.val_tasks:
            print(nEpoch, batch_idx)
            torch.cuda.empty_cache()
            with torch.no_grad():
                vt.test(rot)
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
            self.modular_dict[modk].save_if_needed(nEpoch, batch_idx)


class NekoModularJointTrainingPara(NekoAbstractModularJointTraining):
    def tr_iter(self, nEpoch, batch_idx, sample_batched):
        # torch.autograd.set_detect_anomaly(True)
        # data prepare
        # torch.backends.cudnn.benchmark=True

        zg_start = time.time()

        for modk in self.modular_dict:
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
            self.modular_dict[modk].save_if_needed(nEpoch, batch_idx)


class NekoModularJointTrainingPara2(NekoAbstractModularJointTraining):
    def launch(self, rot, sample_batched, nEpoch, batch_idx):
        l = rot.fp(sample_batched, self.modular_dict, nEpoch, batch_idx)
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
        l = rot.fp(sample_batched, self.modular_dict, nEpoch, batch_idx)
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
            self.modular_dict[modk].save_if_needed(nEpoch, batch_idx)
