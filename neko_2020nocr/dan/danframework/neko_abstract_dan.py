import datetime
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from neko_2020nocr.dan.common.common import display_cfgs, load_dataset, load_all_dataset, Updata_Parameters, \
    generate_optimizer, flatten_label
from neko_2020nocr.dan.utils import LossCounter


class NekoAbstractDan:

    def get_ar_cntr(self, key, case_sensitive):
        return None

    def get_rej_ar_cntr(self, key, case_sensitive):
        return None

    def get_loss_cntr(self, show_interval):
        return LossCounter(show_interval)

    def set_cntrs(self):
        self.train_acc_counter = self.get_ar_cntr('train accuracy: ',
                                                  self.cfgs.dataset_cfgs['te_case_sensitive'])
        self.test_acc_counter = self.get_ar_cntr('\ntest accuracy: ',
                                                 self.cfgs.dataset_cfgs['te_case_sensitive'])
        self.test_rej_counter = self.get_rej_ar_cntr('\ntest rej accuracy: ',
                                                     self.cfgs.dataset_cfgs['te_case_sensitive'])
        self.loss_counter = self.get_loss_cntr(self.cfgs.global_cfgs['show_interval'])

    def setuploss(self):
        self.criterion_CE = nn.CrossEntropyLoss().cuda()

    def load_network(self):
        pass

    def set_up_etc(self):
        pass

    def setup_dataloaders(self):
        try:
            if "dataset_train" in self.cfgs.dataset_cfgs:
                self.train_loader, self.test_loader = load_dataset(self.cfgs, DataLoader)
                self.set_cntrs()
            else:
                self.all_test_loaders = load_all_dataset(self.cfgs, DataLoader)
        except:
            print("no scheduled datasets")

    def setup(self):
        self.model = self.load_network()
        self.setuploss()
        self.optimizers, self.optimizer_schedulers = generate_optimizer(self.cfgs, self.model)
        print('preparing done')
        # --------------------------------
        # prepare tools
        self.set_up_etc()

        pass

    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.cfgs.mkdir(self.cfgs.saving_cfgs['saving_path'])
        self.setup_dataloaders()
        self.setup()
        display_cfgs(self.cfgs, self.model)

        # ---------------------------------

    def test(self, test_loader, model, tools, miter=1000, debug=False, dbgpath=None):
        pass

    def test2g(self, test_loader, model, tools, miter=1000, debug=False, dbgpath=None, protoidx=0):
        pass

    def dump(self, test_loader, model, tools, miter=100, debug=False, dbgpath=None):
        pass

    def fpbp(self, data, label, cased=None):
        pass

    def fpbp2(self, sample_batched):
        data = sample_batched['image']
        label = sample_batched['label']
        if ("cased" in sample_batched):
            return self.fpbp(data, label, sample_batched["cased"])
        return self.fpbp(data, label)

    def runtest(self, miter=1000, debug=False, dbgpath=None, measure_rej=False):
        pass

    def rundump(self, miter=1000, debug=False, dbgpath=None):
        pass

    def dump_chk(self, dbgpath):
        with torch.no_grad():
            self.rundump(self.cfgs.global_cfgs['test_miter'], True, dbgpath)
        exit()

    def show(self):
        self.train_acc_counter.show()
        self.train_acc_counter.clear()

    def tr_iter(self, nEpoch, batch_idx, sample_batched, total_iters):
        # torch.autograd.set_detect_anomaly(True)
        # data prepare

        self.fpbp2(sample_batched)

        for i in range(len(self.model)):
            nn.utils.clip_grad_norm_(self.model[i].parameters(), 20, 2)
        Updata_Parameters(self.optimizers, frozen=[])
        # visualization and saving
        if batch_idx % self.cfgs.global_cfgs['show_interval'] == 0 and batch_idx != 0:
            print(datetime.datetime.now().strftime('%H:%M:%S'))
            loss, terms = self.loss_counter.get_loss_and_terms()
            print('Epoch: {}, Iter: {}/{}, Loss dan: {}'.format(
                nEpoch,
                batch_idx,
                total_iters,
                loss))
            if (len(terms)):
                print(terms)
            self.show()
        if batch_idx % self.cfgs.global_cfgs['test_interval'] == 0 and batch_idx != 0:
            with torch.no_grad():
                self.runtest(self.cfgs.global_cfgs['test_miter'])
        if nEpoch % self.cfgs.saving_cfgs['saving_epoch_interval'] == 0 and \
                batch_idx % self.cfgs.saving_cfgs['saving_iter_interval'] == 0 and \
                batch_idx != 0:
            for i in range(0, len(self.model)):
                torch.save(self.model[i].state_dict(),
                           self.cfgs.saving_cfgs['saving_path'] + 'E{}_I{}-{}_M{}.pth'.format(
                               nEpoch, batch_idx, total_iters, i))

    def run(self, dbgpath=None, measure_rej=False):
        # ---------------------------------
        if self.cfgs.global_cfgs['state'] == 'Test':
            with torch.no_grad():
                self.runtest(self.cfgs.global_cfgs['test_miter'], False, dbgpath, measure_rej=measure_rej)
            return
        # --------------------------------
        total_iters = len(self.train_loader)
        for model in self.model:
            model.train()
        for nEpoch in range(0, self.cfgs.global_cfgs['epoch']):
            for batch_idx, sample_batched in enumerate(self.train_loader):
                self.tr_iter(nEpoch, batch_idx, sample_batched, total_iters)
            Updata_Parameters(self.optimizer_schedulers, frozen=[])
            for i in range(0, len(self.model)):
                torch.save(self.model[i].state_dict(),
                           self.cfgs.saving_cfgs['saving_path'] + 'E{}_M{}.pth'.format(
                               nEpoch, i))
        for model in self.model:
            model.test()
        self.runtest(miter=1000000000000000)
        print("done!")

    def test_all(self, dbgpath=None):
        debug = False
        if (dbgpath):
            debug = True
        retdb = {}
        for ds in self.all_test_loaders:
            if (dbgpath is not None):
                dspath = os.path.join(dbgpath, ds)
            else:
                dspath = None
            test_acc_counter = self.get_ar_cntr(ds, False)
            with torch.no_grad():
                self.test(self.all_test_loaders[ds], self.model, [test_acc_counter,
                                                                  flatten_label,
                                                                  ], miter=999999999, debug=debug, dbgpath=dspath)
            retdb[ds] = test_acc_counter
            # test_acc_counter.show()
        exit()
