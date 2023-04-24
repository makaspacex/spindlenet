# bogomodules are certain combination to the modules, they do not hold parameters
# instead they use whatever armed to the module set.
# Different to routines, they are statically associated to a certain set of modules for speed up.
import torch
from torch.nn import functional as trnf


# some time you cannot have a container. That's life.
# say u have module a b c d.
# A uses [ac] B uses [ab] C uses [ad]...
# There is no better way than simply put a,b,c,d in a big basket.

class prototyper_gen3:
    def __init__(self, args, moddict):
        self.force_proto_shape = args["force_proto_shape"]
        self.capacity = args["capacity"]
        # sometimes we do not need sp protos.
        if (args["sp_proto"] in moddict):
            self.sp = moddict[args["sp_proto"]]
        else:
            self.sp = None
        self.backbone = moddict[args["backbone"]]
        self.cam = moddict[args["cam"]]
        self.device_indicator = "cuda"
        if (args["drop"] is not None):
            self.drop = moddict[args["drop"]]
        else:
            self.drop = None

    def freeze(self):
        self.backbone.model.freeze()
        if (self.sp is not None):
            self.sp.eval()
        self.cam.eval()

    def freeze_bb_bn(self):
        self.backbone.model.freezebn()

    def unfreeze_bb_bn(self):
        self.backbone.model.unfreezebn()

    def unfreeze(self):
        self.backbone.model.unfreeze()
        if (self.sp is not None):
            self.sp.train()
        self.cam.train()

    def cuda(self):
        self.device_indicator = "cuda"

    def proto_engine(self, clips):
        features = self.backbone(clips)
        A = self.cam(features)
        # A=torch.ones_like(A)
        out_emb = (A * features[-1]).sum(-1).sum(-1) / A.sum(-1).sum(-1)
        return out_emb

    def forward(self, normprotos, rot=0, use_sp=True):
        if (len(normprotos) <= self.capacity):
            # pimage=torch.cat(normprotos).to(self.dev_ind.device)
            pimage = torch.cat(normprotos).contiguous().to(self.device_indicator)
            if (self.force_proto_shape is not None and pimage.shape[-1] != self.force_proto_shape):
                pimage = trnf.interpolate(pimage, [self.force_proto_shape, self.force_proto_shape], mode="bilinear")
            if (rot > 0):
                pimage = torch.rot90(pimage, rot, [2, 3])

            if (pimage.shape[1] == 1):
                pimage = pimage.repeat([1, 3, 1, 1])
            if (use_sp):
                spproto, _ = self.sp()
                proto = [spproto, self.proto_engine(pimage)]
            else:
                proto = [self.proto_engine(pimage)]
        else:
            if (use_sp):
                spproto, _ = self.sp()
                proto = [spproto]
            else:
                proto = []
            chunk = self.capacity // 4
            for s in range(0, len(normprotos), chunk):
                pimage = torch.cat(normprotos[s:s + chunk]).contiguous().to(self.device_indicator)
                if (rot > 0):
                    pimage = torch.rot90(pimage, rot, [2, 3])
                if (pimage.shape[1] == 1):
                    pimage = pimage.repeat([1, 3, 1, 1])
                    proto.append(self.proto_engine(pimage))
        allproto = trnf.normalize(torch.cat(proto), dim=1, eps=0.0009)
        if (self.drop):
            allproto = self.drop(allproto)
        return allproto.contiguous()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class prototyper_gen3d(prototyper_gen3):
    def proto_engine(self, clips):
        features = self.backbone(clips)
        A = self.cam([f.detach() for f in features])
        # A=torch.ones_like(A)
        out_emb = (A * features[-1]).sum(-1).sum(-1) / A.sum(-1).sum(-1)
        return out_emb
