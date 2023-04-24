class NekoCtxSubroutineV1head(object):
    def engage(self, module_dict, fsp, csp, logits, lengthl, length, plabel, glabel_flatten, gtdict):
        ctx_logits, compact_ctx_logits = module_dict["ctxmodule"](logits, lengthl, csp, fsp, length.max().item())
        gchoutput, gprdt_prob = module_dict["sampler"].model.decode(ctx_logits, length, fsp, plabel, gtdict)
        gloss_, gterms_ = module_dict["ctxloss"](fsp, ctx_logits, glabel_flatten)
        return gloss_, gchoutput, compact_ctx_logits


class NekoCtxSubroutineV1(object):
    def engage_eval(self, module_dict, fsp, csp, logits, lengthl, length, plabel, gtdict):
        ctx_logits, compact_ctx_logits = module_dict["ctxmodule"](logits, lengthl, csp, fsp, length.max().item())
        gchoutput, gprdt_prob = module_dict["sampler"].model.decode(ctx_logits, length, fsp, plabel, gtdict)
        return ctx_logits, gchoutput, compact_ctx_logits

    def engage(self, module_dict, fsp, csp, logits, lengthl, length, plabel, glabel_flatten, gtdict):
        ctx_logits, gchoutput, compact_ctx_logits = self.engage_eval(module_dict, fsp, csp, logits, lengthl, length,
                                                                     plabel, gtdict)
        gloss_, gterms_ = module_dict["ctxloss"](fsp, ctx_logits, glabel_flatten)
        return gloss_, gchoutput, compact_ctx_logits


class NekoCtxSubroutineV1d(NekoCtxSubroutineV1):

    def engage(self, module_dict, fsp, csp, logits, lengthl, length, plabel, glabel_flatten, gtdict):
        ctx_logits, gchoutput, compact_ctx_logits = self.engage_eval(module_dict, fsp, csp, logits.detach(), lengthl,
                                                                     length, plabel, gtdict)
        gloss_, gterms_ = module_dict["ctxloss"](fsp, ctx_logits, glabel_flatten)
        return gloss_, gchoutput, compact_ctx_logits


class NekoCtxSubroutineV1fb(object):
    def engage(self, module_dict, fsp, csp, logits, lengthl, length, plabel, glabel_flatten, gtdict):
        ctx_logits, compact_ctx_logits = module_dict["ctxmodule"](logits, lengthl, csp, fsp, length.max().item())
        gchoutput, gprdt_prob = module_dict["sampler"].model.decode(ctx_logits, length, fsp, plabel, gtdict)
        gloss_, gterms_ = module_dict["losses"](fsp, ctx_logits, glabel_flatten)
        return gloss_, gchoutput, compact_ctx_logits
