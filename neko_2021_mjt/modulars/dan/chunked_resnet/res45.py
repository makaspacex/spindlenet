import torch
from torch import nn

from neko_2021_mjt.modulars.dan.chunked_resnet.neko_block_fe import (
    make_init_layer_wo_bn,
    make_init_layer_bn,
    make_init_layer_ln,
    InitLayer,
    make_body_layer_wo_bn,
    make_body_layer_bn,
    make_body_layer_norm,
    DanReslayer,
)
from neko_sdk.AOF.neko_lens import NekoLens


# Dan config.
def res45tpt_wo_bn(inpch, oupch, strides, frac=1, ochs=None):
    retlayers = {}
    blkcnt = [None, 3, 4, 6, 6, 3]
    if ochs is None:
        ochs = [
            int(32 * frac),
            int(32 * frac),
            int(64 * frac),
            int(128 * frac),
            int(256 * frac),
            oupch,
        ]
    retlayers["0"] = make_init_layer_wo_bn(inpch[0], ochs[0], strides[0])
    retlayers["1"] = make_body_layer_wo_bn(ochs[0], blkcnt[1], ochs[1], 1, strides[1])
    retlayers["2"] = make_body_layer_wo_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2])
    retlayers["3"] = make_body_layer_wo_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3])
    retlayers["4"] = make_body_layer_wo_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4])
    retlayers["5"] = make_body_layer_wo_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5])
    return retlayers


def res45_wo_bn(inpch, oupch, strides, frac=1, ochs=None):
    retlayers = {}
    blkcnt = [None, 3, 4, 6, 6, 3]
    if ochs is None:
        ochs = [
            int(32 * frac),
            int(32 * frac),
            int(64 * frac),
            int(128 * frac),
            int(256 * frac),
            oupch,
        ]
    retlayers["0"] = make_init_layer_wo_bn(inpch[0], ochs[0], strides[0])
    retlayers["1"] = make_body_layer_wo_bn(ochs[0], blkcnt[1], ochs[1], 1, strides[1])
    retlayers["2"] = make_body_layer_wo_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2])
    retlayers["3"] = make_body_layer_wo_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3])
    retlayers["4"] = make_body_layer_wo_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4])
    retlayers["5"] = make_body_layer_wo_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5])
    return retlayers

def res45_wo_bn_wide_64(inpch, oupch, strides, frac=1, ochs=None):
    retlayers = {}
    blkcnt = [None, 3, 4, 6, 6, 3]
    if ochs is None:
        ochs = [
            int(32 * frac),
            int(64 * frac),
            int(64 * frac),
            int(128 * frac),
            int(256 * frac),
            oupch,
        ]
    retlayers["0"] = make_init_layer_wo_bn(inpch[0], ochs[0], strides[0])
    retlayers["1"] = make_body_layer_wo_bn(ochs[0], blkcnt[1], ochs[1], 1, strides[1])
    retlayers["2"] = make_body_layer_wo_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2])
    retlayers["3"] = make_body_layer_wo_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3])
    retlayers["4"] = make_body_layer_wo_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4])
    retlayers["5"] = make_body_layer_wo_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5])
    return retlayers


def res45_bn(inpch, oupch, strides, frac=1, ochs=None):
    blkcnt = [None, 3, 4, 6, 6, 3]
    if ochs is None:
        ochs = [
            int(32 * frac),
            int(32 * frac),
            int(64 * frac),
            int(128 * frac),
            int(256 * frac),
            oupch,
        ]
    retlayers = {}
    retlayers["0"] = make_init_layer_bn(ochs[0])
    retlayers["1"] = make_body_layer_bn(ochs[0], blkcnt[1], ochs[1], 1, strides[1])
    retlayers["2"] = make_body_layer_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2])
    retlayers["3"] = make_body_layer_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3])
    retlayers["4"] = make_body_layer_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4])
    retlayers["5"] = make_body_layer_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5])
    return retlayers


def res45_ln(inpch, oupch, strides, frac=1, ochs=None):
    blkcnt = [None, 3, 4, 6, 6, 3]
    if ochs is None:
        ochs = [
            int(32 * frac),
            int(32 * frac),
            int(64 * frac),
            int(128 * frac),
            int(256 * frac),
            oupch,
        ]
    retlayers = {}
    retlayers["0"] = make_init_layer_ln(ochs[0])
    retlayers["1"] = make_body_layer_norm(ochs[0], blkcnt[1], ochs[1], 1, strides[1])
    retlayers["2"] = make_body_layer_norm(ochs[1], blkcnt[2], ochs[2], 1, strides[2])
    retlayers["3"] = make_body_layer_norm(ochs[2], blkcnt[3], ochs[3], 1, strides[3])
    retlayers["4"] = make_body_layer_norm(ochs[3], blkcnt[4], ochs[4], 1, strides[4])
    retlayers["5"] = make_body_layer_norm(ochs[4], blkcnt[5], ochs[5], 1, strides[5])
    return retlayers


# OSOCR config. Seems they have much better perf due to the heavier layout
# The called the method ``pami'' www
def res45ptpt_wo_bn(inpch, oupch, strides, frac=1, ochs=None):
    retlayers = {}
    blkcnt = [None, 3, 4, 6, 6, 3]
    if ochs is None:
        ochs = [
            int(32 * frac),
            int(32 * frac),
            int(64 * frac),
            int(128 * frac),
            int(256 * frac),
            oupch,
        ]
    retlayers["0"] = make_init_layer_wo_bn(inpch[0], ochs[0], strides[0])
    retlayers["1"] = make_body_layer_wo_bn(ochs[0], blkcnt[1], ochs[1], 1, strides[1])
    retlayers["2"] = make_body_layer_wo_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2])
    retlayers["3"] = make_body_layer_wo_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3])
    retlayers["4"] = make_body_layer_wo_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4])
    retlayers["5"] = make_body_layer_wo_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5])
    return retlayers


def res45p_wo_bn(inpch, oupch, strides, frac=1, ochs=None):
    retlayers = {}
    blkcnt = [None, 3, 4, 6, 6, 3]
    if ochs is None:
        ochs = [
            int(32 * frac),
            int(64 * frac),
            int(128 * frac),
            int(256 * frac),
            int(512 * frac),
            oupch,
        ]
    retlayers["0"] = make_init_layer_wo_bn(inpch[0], ochs[0], strides[0])
    retlayers["1"] = make_body_layer_wo_bn(ochs[0], blkcnt[1], ochs[1], 1, strides[1])
    retlayers["2"] = make_body_layer_wo_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2])
    retlayers["3"] = make_body_layer_wo_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3])
    retlayers["4"] = make_body_layer_wo_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4])
    retlayers["5"] = make_body_layer_wo_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5])
    return retlayers


def res45p_bn(inpch, oupch, strides, frac=1, ochs=None):
    blkcnt = [None, 3, 4, 6, 6, 3]
    if ochs is None:
        ochs = [
            int(32 * frac),
            int(64 * frac),
            int(128 * frac),
            int(256 * frac),
            int(512 * frac),
            oupch,
        ]
    retlayers = {}
    retlayers["0"] = make_init_layer_bn(ochs[0])
    retlayers["1"] = make_body_layer_bn(ochs[0], blkcnt[1], ochs[1], 1, strides[1])
    retlayers["2"] = make_body_layer_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2])
    retlayers["3"] = make_body_layer_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3])
    retlayers["4"] = make_body_layer_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4])
    retlayers["5"] = make_body_layer_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5])
    return retlayers


class Res45Net:
    def __init__(self, layer_dict, bn_dict):
        self.init_layer = InitLayer(layer_dict["0"], bn_dict["0"])
        self.res_layer1 = DanReslayer(layer_dict["1"], bn_dict["1"])
        self.res_layer2 = DanReslayer(layer_dict["2"], bn_dict["2"])
        self.res_layer3 = DanReslayer(layer_dict["3"], bn_dict["3"])
        self.res_layer4 = DanReslayer(layer_dict["4"], bn_dict["4"])
        self.res_layer5 = DanReslayer(layer_dict["5"], bn_dict["5"])

    def __call__(self, x):
        ret = []
        x = self.init_layer(x.contiguous())
        x = self.res_layer1(x)
        x = self.res_layer2(x)
        ret.append(x)
        x = self.res_layer3(x)
        x = self.res_layer4(x)
        ret.append(x)
        x = self.res_layer5(x)
        ret.append(x)
        return ret


class Res45NetOrig:
    def cuda(self):
        pass

    def __init__(self, layer_dict, bn_dict):
        self.init_layer = InitLayer(layer_dict["0"], bn_dict["0"])
        self.res_layer1 = DanReslayer(layer_dict["1"], bn_dict["1"])
        self.res_layer2 = DanReslayer(layer_dict["2"], bn_dict["2"])
        self.res_layer3 = DanReslayer(layer_dict["3"], bn_dict["3"])
        self.res_layer4 = DanReslayer(layer_dict["4"], bn_dict["4"])
        self.res_layer5 = DanReslayer(layer_dict["5"], bn_dict["5"])

    def __call__(self, x):
        ret = []
        x = self.init_layer(x.contiguous())
        tmp_shape = x.size()[2:]
        x = self.res_layer1(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x)
        x = self.res_layer2(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x)
        x = self.res_layer3(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x)
        x = self.res_layer4(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x)
        x = self.res_layer5(x)
        ret.append(x)
        return ret


class Res45NetTpt:
    def cuda(self):
        pass

    def __init__(self, layer_dict, bn_dict, lens):
        self.init_layer = InitLayer(layer_dict["0"], bn_dict["0"])
        self.lens = lens
        self.res_layer1 = DanReslayer(layer_dict["1"], bn_dict["1"])
        self.res_layer2 = DanReslayer(layer_dict["2"], bn_dict["2"])
        self.res_layer3 = DanReslayer(layer_dict["3"], bn_dict["3"])
        self.res_layer4 = DanReslayer(layer_dict["4"], bn_dict["4"])
        self.res_layer5 = DanReslayer(layer_dict["5"], bn_dict["5"])

    def __call__(self, x):
        ret = []
        x = self.init_layer(x.contiguous())
        x, _ = self.lens(x)
        tmp_shape = x.size()[2:]
        x = self.res_layer1(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x)
        x = self.res_layer2(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x)
        x = self.res_layer3(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x)
        x = self.res_layer4(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x)
        x = self.res_layer5(x)
        ret.append(x)
        return ret


class Res45NetPtpt:
    def cuda(self):
        pass

    def __init__(self, layer_dict, bn_dict, lens):
        self.init_layer = InitLayer(layer_dict["0"], bn_dict["0"])
        self.lens = lens
        self.res_layer1 = DanReslayer(layer_dict["1"], bn_dict["1"])
        self.res_layer2 = DanReslayer(layer_dict["2"], bn_dict["2"])
        self.res_layer3 = DanReslayer(layer_dict["3"], bn_dict["3"])
        self.res_layer4 = DanReslayer(layer_dict["4"], bn_dict["4"])
        self.res_layer5 = DanReslayer(layer_dict["5"], bn_dict["5"])

    def __call__(self, x):
        ret = []
        x = self.init_layer(x.contiguous())
        x, _ = self.lens(x)
        tmp_shape = x.size()[2:]
        x = self.res_layer1(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x)
        x = self.res_layer2(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x)
        x = self.res_layer3(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x)
        x = self.res_layer4(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x)
        x = self.res_layer5(x)
        ret.append(x)
        return ret


# so this thing keeps the modules and
class NekoR45Binorm(nn.Module):
    def setup_modules(self, mdict, prefix):
        for k in mdict:
            if type(mdict[k]) is dict:
                self.setup_modules(mdict[k], prefix + "_" + k)
            else:
                self.add_module(prefix + "_" + k, mdict[k])

    def setup_bn_modules(self, mdict, prefix):
        for k in mdict:
            if type(mdict[k]) is dict:
                self.setup_bn_modules(mdict[k], prefix + "_" + k)
            else:
                id = len(self.bns)
                if prefix not in self.bn_dict:
                    self.bn_dict[prefix] = []
                self.bn_dict[prefix].append(id)
                self.add_module(prefix + "_" + k, mdict[k])
                self.bns.append(mdict[k])

    def freezebnprefix(self, prefix):
        for i in self.bn_dict[prefix]:
            self.bns[i].eval()

    def unfreezebnprefix(self, prefix):
        for i in self.bn_dict[prefix]:
            self.bns[i].train()

    def freezebn(self):
        for i in self.bns:
            i.eval()

    def unfreezebn(self):
        for i in self.bns:
            i.train()

    def __init__(
        self,
        strides,
        compress_layer,
        input_shape,
        bogo_names,
        bn_names,
        hardness=2,
        oupch=512,
        expf=1,
    ):
        super(NekoR45Binorm, self).__init__()
        self.bogo_modules = {}
        self.bn_dict = {}
        layers = res45_wo_bn(
            input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], 1
        )
        self.setup_modules(layers, "shared_fe")
        self.bns = []
        for i in range(len(bogo_names)):
            name = bogo_names[i]
            bn_name = bn_names[i]
            bns = res45_bn(
                input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], 1
            )
            self.bogo_modules[name] = Res45Net(layers, bns)
            self.setup_bn_modules(bns, bn_name)

    def forward(self, input, debug=False):
        # This won't work as this is just a holder
        exit(9)

    def Iwantshapes(self):
        pseudo_input = torch.rand(
            1, self.input_shape[0], self.input_shape[1], self.input_shape[2]
        )
        features, grio = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]


class NekoR45BinormOrig(nn.Module):
    def freezebnprefix(self, prefix):
        for i in self.named_bn_dicts[prefix]:
            self.bns[i].eval()

    def unfreezebnprefix(self, prefix):
        for i in self.named_bn_dicts[prefix]:
            self.bns[i].train()

    def setup_bn_modules(self, mdict, prefix, gprefix):
        if gprefix not in self.named_bn_dicts:
            self.named_bn_dicts[gprefix] = []

        for k in mdict:
            if type(mdict[k]) is dict:
                self.setup_bn_modules(mdict[k], prefix + "_" + k, gprefix)
            else:
                self.add_module(prefix + "_" + k, mdict[k])
                self.named_bn_dicts[gprefix].append(len(self.bns))
                self.bns.append(mdict[k])

    def freezebn(self):
        for i in self.bns:
            i.eval()

    def unfreezebn(self):
        for i in self.bns:
            i.train()

    def setup_modules(self, mdict, prefix):
        for k in mdict:
            if type(mdict[k]) is dict:
                self.setup_modules(mdict[k], prefix + "_" + k)
            else:
                self.add_module(prefix + "_" + k, mdict[k])

    def __init__(
        self,
        strides,
        compress_layer,
        input_shape,
        bogo_names,
        bn_names,
        hardness=2,
        oupch=512,
        expf=1,
        ch_overid_num=None
    ):
        super(NekoR45BinormOrig, self).__init__()
        self.bogo_modules = {}
        layers = res45_wo_bn(
            input_shape,
            oupch,
            [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
            frac=expf,
            ochs=ch_overid_num
        )
        self.setup_modules(layers, "shared_fe")
        self.bns = []
        self.named_bn_dicts = {}
        for i in range(len(bogo_names)):
            name = bogo_names[i]
            bn_name = bn_names[i]
            bns = res45_bn(
                input_shape,
                oupch,
                [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
                frac=expf,
                ochs= ch_overid_num
            )
            self.bogo_modules[name] = Res45NetOrig(layers, bns)
            self.setup_bn_modules(bns, bn_name, bn_name)


class NekoR45BinormLayerNorm(nn.Module):
    def freezebnprefix(self, prefix):
        for i in self.named_bn_dicts[prefix]:
            self.bns[i].eval()

    def unfreezebnprefix(self, prefix):
        for i in self.named_bn_dicts[prefix]:
            self.bns[i].train()

    def setup_bn_modules(self, mdict, prefix, gprefix):
        if gprefix not in self.named_bn_dicts:
            self.named_bn_dicts[gprefix] = []

        for k in mdict:
            if type(mdict[k]) is dict:
                self.setup_bn_modules(mdict[k], prefix + "_" + k, gprefix)
            else:
                self.add_module(prefix + "_" + k, mdict[k])
                self.named_bn_dicts[gprefix].append(len(self.bns))
                self.bns.append(mdict[k])

    def freezebn(self):
        for i in self.bns:
            i.eval()

    def unfreezebn(self):
        for i in self.bns:
            i.train()

    def setup_modules(self, mdict, prefix):
        for k in mdict:
            if type(mdict[k]) is dict:
                self.setup_modules(mdict[k], prefix + "_" + k)
            else:
                self.add_module(prefix + "_" + k, mdict[k])

    def __init__(
        self,
        strides,
        compress_layer,
        input_shape,
        bogo_names,
        bn_names,
        hardness=2,
        oupch=512,
        expf=1,
        ch_overid_num=None
    ):
        super(NekoR45BinormOrig, self).__init__()
        self.bogo_modules = {}
        layers = res45_wo_bn(
            input_shape,
            oupch,
            [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
            frac=expf,
            ochs=ch_overid_num
        )
        self.setup_modules(layers, "shared_fe")
        self.bns = []
        self.named_bn_dicts = {}
        for i in range(len(bogo_names)):
            name = bogo_names[i]
            bn_name = bn_names[i]
            bns = res45_ln(
                input_shape,
                oupch,
                [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
                frac=expf,
                ochs= ch_overid_num
            )
            self.bogo_modules[name] = Res45NetOrig(layers, bns)
            self.setup_bn_modules(bns, bn_name, bn_name)


class NekoR45BinormTpt(nn.Module):
    def freezebnprefix(self, prefix):
        for i in self.named_bn_dicts[prefix]:
            self.bns[i].eval()

    def unfreezebnprefix(self, prefix):
        for i in self.named_bn_dicts[prefix]:
            self.bns[i].train()

    def setup_bn_modules(self, mdict, prefix, gprefix):
        if gprefix not in self.named_bn_dicts:
            self.named_bn_dicts[gprefix] = []

        for k in mdict:
            if type(mdict[k]) is dict:
                self.setup_bn_modules(mdict[k], prefix + "_" + k, gprefix)
            else:
                self.add_module(prefix + "_" + k, mdict[k])
                self.named_bn_dicts[gprefix].append(len(self.bns))
                self.bns.append(mdict[k])

    def freezebn(self):
        for i in self.bns:
            i.eval()

    def unfreezebn(self):
        for i in self.bns:
            i.train()

    def setup_modules(self, mdict, prefix):
        for k in mdict:
            if type(mdict[k]) is dict:
                self.setup_modules(mdict[k], prefix + "_" + k)
            else:
                self.add_module(prefix + "_" + k, mdict[k])

    def __init__(
        self,
        strides,
        compress_layer,
        input_shape,
        bogo_names,
        bn_names,
        hardness=2,
        oupch=512,
        expf=1,
    ):
        super(NekoR45BinormTpt, self).__init__()
        self.bogo_modules = {}
        layers = res45_wo_bn(
            input_shape,
            oupch,
            [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
            frac=expf,
        )
        self.setup_modules(layers, "shared_fe")
        self.bns = []
        self.tpt = NekoLens(int(32 * expf), 1, 1, hardness)
        self.named_bn_dicts = {}
        for i in range(len(bogo_names)):
            name = bogo_names[i]
            bn_name = bn_names[i]
            bns = res45_bn(
                input_shape,
                oupch,
                [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
                frac=expf,
            )
            self.bogo_modules[name] = Res45NetOrig(layers, bns)
            self.setup_bn_modules(bns, bn_name, bn_name)

    def forward(self, input, debug=False):
        # This won't work as this is just a holder
        exit(9)

    def Iwantshapes(self):
        pseudo_input = torch.rand(
            1, self.input_shape[0], self.input_shape[1], self.input_shape[2]
        )
        features, grio = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]


class NekoR45BinormPtpt(nn.Module):
    def freezebnprefix(self, prefix):
        for i in self.named_bn_dicts[prefix]:
            self.bns[i].eval()

    def unfreezebnprefix(self, prefix):
        for i in self.named_bn_dicts[prefix]:
            self.bns[i].train()

    def setup_bn_modules(self, mdict, prefix, gprefix):
        if gprefix not in self.named_bn_dicts:
            self.named_bn_dicts[gprefix] = []

        for k in mdict:
            if type(mdict[k]) is dict:
                self.setup_bn_modules(mdict[k], prefix + "_" + k, gprefix)
            else:
                self.add_module(prefix + "_" + k, mdict[k])
                self.named_bn_dicts[gprefix].append(len(self.bns))
                self.bns.append(mdict[k])

    def freezebn(self):
        for i in self.bns:
            i.eval()

    def unfreezebn(self):
        for i in self.bns:
            i.train()

    def setup_modules(self, mdict, prefix):
        for k in mdict:
            if type(mdict[k]) is dict:
                self.setup_modules(mdict[k], prefix + "_" + k)
            else:
                self.add_module(prefix + "_" + k, mdict[k])

    def __init__(
        self,
        strides,
        compress_layer,
        input_shape,
        bogo_names,
        bn_names,
        hardness=2,
        oupch=512,
        expf=1,
    ):
        super(NekoR45BinormPtpt, self).__init__()
        self.bogo_modules = {}
        layers = res45p_wo_bn(
            input_shape,
            oupch,
            [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
            frac=expf,
        )
        self.setup_modules(layers, "shared_fe")
        self.bns = []
        self.tpt = NekoLens(int(32 * expf), 1, 1, hardness)
        self.named_bn_dicts = {}
        for i in range(len(bogo_names)):
            name = bogo_names[i]
            bn_name = bn_names[i]
            bns = res45p_bn(
                input_shape,
                oupch,
                [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
                frac=expf,
            )
            self.bogo_modules[name] = Res45NetOrig(layers, bns)
            self.setup_bn_modules(bns, bn_name, bn_name)

    def forward(self, input, debug=False):
        # This won't work as this is just a holder
        exit(9)

    def Iwantshapes(self):
        pseudo_input = torch.rand(
            1, self.input_shape[0], self.input_shape[1], self.input_shape[2]
        )
        features, grio = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]


class NekoR45BinormHeavyHead(nn.Module):
    def setup_modules(self, mdict, prefix):
        for k in mdict:
            if type(mdict[k]) is dict:
                self.setup_modules(mdict[k], prefix + "_" + k)
            else:
                self.add_module(prefix + "_" + k, mdict[k])

    def __init__(
        self,
        strides,
        compress_layer,
        input_shape,
        bogo_names,
        bn_names,
        hardness=2,
        oupch=512,
        expf=1,
    ):
        super(NekoR45BinormHeavyHead, self).__init__()
        self.bogo_modules = {}
        ochs = [
            int(64 * expf),
            int(64 * expf),
            int(64 * expf),
            int(128 * expf),
            int(256 * expf),
            oupch,
        ]

        layers = res45_wo_bn(
            input_shape,
            oupch,
            [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
            1,
            ochs=ochs,
        )
        self.setup_modules(layers, "shared_fe")
        for i in range(len(bogo_names)):
            name = bogo_names[i]
            bn_name = bn_names[i]
            bns = res45_bn(
                input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], 1
            )
            self.bogo_modules[name] = Res45NetOrig(layers, bns)
            self.setup_modules(bns, bn_name)

    def forward(self, input, debug=False):
        # This won't work as this is just a holder
        exit(9)

    def Iwantshapes(self):
        pseudo_input = torch.rand(
            1, self.input_shape[0], self.input_shape[1], self.input_shape[2]
        )
        features, grio = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]


# 兼容旧模型
neko_r45_binorm_orig = NekoR45BinormOrig
res45_net_orig = Res45NetOrig
if __name__ == "__main__":
    layers = res45_wo_bn(3, 512, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], 1)
    bns = res45_bn(3, 512, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], 1)
    a = Res45Net(layers, bns)
    t = torch.rand([1, 3, 32, 128])
    r = a(t)
    pass
