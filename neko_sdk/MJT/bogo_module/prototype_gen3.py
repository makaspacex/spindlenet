# bogomodules are certain combination to the modules, they do not hold parameters
# instead they use whatever armed to the module set.
# Different to routines, they are statically associated to a certain set of modules for speed up.
import torch
from torch.nn import functional as trnf


# some time you cannot have a container. That's life.
# say u have module a b c d.
# A uses [ac] B uses [ab] C uses [ad]...
# There is no better way than simply put a,b,c,d in a big basket.
class gen3_object_to_feat_abstract:
    def proto_engine(self, clips):
        features = self.backbone(clips)
        if (self.detached_ga):
            A = self.cam([f.detach() for f in features])
        else:
            A = self.cam(features)
        # A=torch.ones_like(A)
        out_emb = (A * features[-1]).sum(-1).sum(-1) / A.sum(-1).sum(-1)
        return out_emb

    def proto_engine_debug(self, clips):
        features = self.backbone(clips)
        if (self.detached_ga):
            A = self.cam([f.detach() for f in features])
        else:
            A = self.cam(features)

        # A=torch.ones_like(A)
        out_emb = (A * features[-1]).sum(-1).sum(-1) / A.sum(-1).sum(-1)
        return out_emb, A

    def detach(self):
        self.backbone.detach()
        self.cam.detach()

    def attach(self):
        self.backbone.model.attach()
        self.cam.attach()

    def __init__(self, args, moddict):
        self.detached_ga = args["detached_ga"]
        self.backbone = moddict[args["backbone"]]
        self.cam = moddict[args["cam"]]


# stand can only call the modules.
# only (bogo)modules can change their statues like training, evaluation, accpetance of gradient.
class PrototyperGen3Stand(gen3_object_to_feat_abstract):
    def __init__(self, args, moddict):
        super(PrototyperGen3Stand, self).__init__(args, moddict)
        self.detached_ga = args["detached_ga"]
        self.force_proto_shape = args["force_proto_shape"]
        self.capacity = args["capacity"]
        # sometimes we do not need sp protos.
        if (args["sp_proto"] in moddict):
            self.sp = moddict[args["sp_proto"]]
        else:
            self.sp = None
        if ("device" not in args):
            self.device_indicator = "cuda"
        else:
            self.device_indicator = args["device"]
        if (args["drop"]):
            if (args["drop"] == "NEP_reuse_NEP"):
                self.drop = self.backbone.model.model.dropper
            else:
                self.drop = moddict[args["drop"]]
        else:
            self.drop = None

    def forward_stub(self, normprotos, rot=0, engine=None):
        # pimage=torch.cat(normprotos).to(self.dev_ind.device)
        pimage = torch.cat(normprotos).contiguous().to(self.device_indicator)
        # print(self.device_indicator)
        if (self.force_proto_shape is not None and pimage.shape[-1] != self.force_proto_shape):
            pimage = trnf.interpolate(pimage, [self.force_proto_shape, self.force_proto_shape], mode="bilinear")
        if (rot > 0):
            pimage = torch.rot90(pimage, rot, [2, 3])

        if (pimage.shape[1] == 1):
            pimage = pimage.repeat([1, 3, 1, 1])
        return engine(pimage)

    def forward(self, normprotos, rot=0, use_sp=True):
        if (len(normprotos) <= self.capacity):
            if (use_sp):
                spproto, _ = self.sp()
                proto = [spproto, self.forward_stub(normprotos, rot, engine=self.proto_engine)]
            else:
                proto = [self.forward_stub(normprotos, rot, engine=self.proto_engine)]
        else:
            if (use_sp):
                spproto, _ = self.sp()
                proto = [spproto]
            else:
                proto = []
            chunk = self.capacity // 4
            for s in range(0, len(normprotos), chunk):
                batchp = self.forward_stub(normprotos[s:s + chunk], rot, engine=self.proto_engine)
                proto.append(batchp)
        allproto = torch.cat(proto).contiguous()
        if (self.drop):
            allproto = self.drop(allproto)
        allproto = trnf.normalize(allproto, dim=-1, eps=0.0009)

        return allproto

    def forward_debug(self, normprotos, rot=0, use_sp=True):
        if (use_sp):
            spproto, _ = self.sp()
            proto = [spproto]
        else:
            proto = []
        atts = []
        if (len(normprotos) <= self.capacity):
            p, a = self.forward_stub(normprotos, rot, engine=self.proto_engine_debug)
            proto.append(p)
            atts.append(a.detach().cpu())
        else:
            chunk = self.capacity // 4
            for s in range(0, len(normprotos), chunk):
                batchp, batcha = self.forward_stub(normprotos[s:s + chunk], rot, engine=self.proto_engine_debug)
                proto.append(batchp)
                atts.append(batcha.detach().cpu())
        allproto = torch.cat(proto).contiguous()
        if (self.drop):
            allproto = self.drop(allproto)
        allatt = torch.cat(atts)
        allproto = trnf.normalize(allproto, dim=-1, eps=0.0009)
        # allprotoe,allatte=self.forward_debug_err(normprotos,rot,use_sp)
        return allproto.contiguous(), allatt

    def forward_debug_err(self, normprotos, rot=0, use_sp=True):
        if (len(normprotos) <= self.capacity):
            if (use_sp):
                spproto, _ = self.sp()
                p, a = self.forward_stub(normprotos, rot, engine=self.proto_engine_debug)
                proto = [spproto, p]
                atts = [a.detach().cpu()]
            else:
                p, a = self.forward_stub(normprotos, rot, engine=self.proto_engine_debug)
                proto = [p]
                atts = [a.detach().cpu()]
        else:
            if (use_sp):
                spproto, _ = self.sp()
                proto = [spproto]
            else:
                proto = []
            atts = []
            chunk = self.capacity // 4
            for s in range(0, len(normprotos), chunk):
                batchp, batcha = self.forward_stub(normprotos[s:s + chunk], rot, engine=self.proto_engine_debug)

                proto.append(batchp)
                atts.append(batcha.detach().cpu())
        allproto = torch.cat(proto)
        if (self.drop):
            allproto = self.drop(allproto)
        allproto = trnf.normalize(allproto, dim=-1, eps=0.0009)
        allatt = torch.cat(atts)

        return allproto.contiguous(), allatt

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class PrototyperGen3:
    def get_stand(self, args, moddict):
        return PrototyperGen3Stand(args, moddict)

    # The info are kept to later replicate stands.
    # The api is optimized to support multi-gpu training
    # however the users are responsible to avoid race condition when trying to freeze and unfreeze bn and etc.
    # we ourself do not use multi-gpu setup since we do not currently benefit from multi-gpu training, but since someone requested...
    def __init__(self, args, moddict):
        self.detached_ga = args["detached_ga"]
        self.force_proto_shape = args["force_proto_shape"]
        self.capacity = args["capacity"]
        # sometimes we do not need sp protos.
        if (args["sp_proto"] in moddict):
            self.sp = moddict[args["sp_proto"]]
        else:
            self.sp = None
        self.backbone = moddict[args["backbone"]]
        self.cam = moddict[args["cam"]]
        if ("device" not in args):
            self.device_indicator = "cuda"
        else:
            self.device_indicator = args["device"]
        if (args["drop"]):
            if (args["drop"] == "NEP_reuse_NEP"):
                self.drop = self.backbone.model.model.dropper
            else:
                self.drop = moddict[args["drop"]]
        else:
            self.drop = None
        self.stand = self.get_stand(args, moddict)

    def __call__(self, *args, **kwargs):
        return self.stand(*args, **kwargs)

    def forward_debug(self, *args, **kwargs):
        return self.stand.forward_debug(*args, **kwargs)

    def replicate(self, devices):
        moddicts = [{} for d in devices]
        args = [{} for a in devices]
        self.stands = []
        for did in range(len(devices)):
            args[did]["capacity"] = self.capacity
            args[did]["force_proto_shape"] = self.force_proto_shape
            args[did]["device"] = devices[did]

            moddicts[did]["backbone"] = self.backbone.stands[did]
            args[did]["backbone"] = "backbone"

            moddicts[did]["cam"] = self.cam.stands[did]
            args[did]["cam"] = "cam"

            if (self.sp):
                moddicts[did]["sp_proto"] = self.cam.stands[did]
                args[did]["sp_proto"] = "sp_proto"
            else:
                args[did]["sp_proto"] = "TopNEP"
            if (self.drop):
                moddicts[did]["drop"] = self.cam.stands[did]
                args[did]["drop"] = "drop"
            else:
                args[did]["drop"] = None

            self.stands.append(self.get_stand(args[did], moddicts[did]))
        return self.stands

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


class PrototyperGen3MaskedStand(PrototyperGen3Stand):
    def proto_engine(self, clips):
        features, _ = self.backbone(clips)
        if (self.detached_ga):
            A = self.cam([f.detach() for f in features])
        else:
            A = self.cam(features)
        # A=torch.ones_like(A)
        out_emb = (A * features[-1]).sum(-1).sum(-1) / A.sum(-1).sum(-1)
        return out_emb

    def proto_engine_debug(self, clips):
        features, _ = self.backbone(clips)
        if (self.detached_ga):
            A = self.cam([f.detach() for f in features])
        else:
            A = self.cam(features)  # A=torch.ones_like(A)
        out_emb = (A * features[-1]).sum(-1).sum(-1) / A.sum(-1).sum(-1)
        return out_emb, A


class PrototyperGen3Masked(PrototyperGen3):

    def get_stand(self, args, moddict):
        return PrototyperGen3MaskedStand(args, moddict)


class PrototyperGen3pStand(PrototyperGen3Stand):
    def proto_engine(self, clips):
        features = self.backbone(clips)
        if (self.detached_ga):
            A = self.cam([f.detach() for f in features])
        else:
            A = self.cam(features)
        # A=torch.ones_like(A)
        N, C, H, W = features[-1].shape
        out_emb = (A.unsqueeze(2) * features[-1].reshape(N, A.shape[1], -1, W, H)).sum(-1).sum(-1) / A.sum(-1).sum(
            -1).unsqueeze(-1)
        return out_emb

    def proto_engine_debug(self, clips):
        features = self.backbone(clips)
        A, b = self.cam(features, debug=True)
        # A=torch.ones_like(A)
        N, C, H, W = features[-1].shape
        out_emb = (A.unsqueeze(2) * features[-1].reshape(N, A.shape[1], -1, W, H)).sum(-1).sum(-1) / A.sum(-1).sum(
            -1).unsqueeze(-1)
        return out_emb, torch.cat([A, b], dim=1)

    def forward(self, normprotos, rot=0, use_sp=True):
        if (len(normprotos) <= self.capacity):
            # pimage=torch.cat(normprotos).to(self.dev_ind.device)
            pimage = torch.cat(normprotos).contiguous().to(self.device_indicator)
            # print(self.device_indicator)
            if (self.force_proto_shape is not None and pimage.shape[-1] != self.force_proto_shape):
                pimage = trnf.interpolate(pimage, [self.force_proto_shape, self.force_proto_shape], mode="bilinear")
            if (rot > 0):
                pimage = torch.rot90(pimage, rot, [2, 3])

            if (pimage.shape[1] == 1):
                pimage = pimage.repeat([1, 3, 1, 1])
            if (use_sp):
                spproto, _ = self.sp()
                rpr = self.proto_engine(pimage)
                proto = [spproto.reshape(spproto.shape[0], rpr.shape[1], -1), rpr]
            else:
                proto = [self.proto_engine(pimage)]
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
            if (use_sp):
                spproto, _ = self.sp()
                proto = [spproto.reshape([spproto.shape[0], -1, proto[0].shape[-1]]), *proto]
        allproto = torch.cat(proto)
        if (self.drop):
            allproto = self.drop(allproto)
        allproto = trnf.normalize(allproto, dim=-1, eps=0.0009)

        return allproto.contiguous()


class PrototyperGen3p(PrototyperGen3):
    def get_stand(self, args, moddict):
        return PrototyperGen3pStand(args, moddict)


class PrototyperGen3peStand(PrototyperGen3pStand):
    def proto_engine(self, clips):
        features = self.backbone(clips)
        if (self.detached_ga):
            A = self.cam([f.detach() for f in features])
        else:
            A = self.cam(features)
        # A=torch.ones_like(A)
        N, C, H, W = features[-1].shape
        out_emb = (A.unsqueeze(2) * features[-1].reshape(N, 1, -1, W, H)).sum(-1).sum(-1) / A.sum(-1).sum(-1).unsqueeze(
            -1)
        return out_emb

    def proto_engine_debug(self, clips):
        features = self.backbone(clips)
        if (self.detached_ga):
            A = self.cam([f.detach() for f in features])
        else:
            A = self.cam(features)
        # A=torch.ones_like(A)
        N, C, H, W = features[-1].shape
        out_emb = (A.unsqueeze(2) * features[-1].reshape(N, 1, -1, W, H)).sum(-1).sum(-1) / A.sum(-1).sum(-1).unsqueeze(
            -1)
        return out_emb, A


class PrototyperGen3pe(PrototyperGen3):
    def get_stand(self, args, moddict):
        return PrototyperGen3peStand(args, moddict)


class PrototyperGen3petStand(PrototyperGen3peStand):
    def proto_engine(self, clips):
        features_ffn, features_att = self.backbone(clips)
        # detaching is now up to the GA itself.
        A = self.cam(features_att)
        # A=torch.ones_like(A)
        N, C, H, W = features_ffn[-1].shape
        out_emb = (A.unsqueeze(2) * features_ffn[-1].reshape(N, 1, -1, W, H)).sum(-1).sum(-1) / A.sum(-1).sum(
            -1).unsqueeze(-1)
        return out_emb


class PrototyperGen3pet(PrototyperGen3):
    def get_stand(self, args, moddict):
        return PrototyperGen3petStand(args, moddict)
