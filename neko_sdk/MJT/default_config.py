from torch import optim


def get_default_optim(params, lr=1.0, sched_override=None):
    optimizer = optim.Adadelta(params, lr=lr)
    if (sched_override is None):
        optimizer_sched = optim.lr_scheduler.MultiStepLR(optimizer, [3, 5], 0.1)
    else:
        optimizer_sched = sched_override["engine"](optimizer, **sched_override["param"])
    return optimizer, optimizer_sched


def get_default_model(mod, args, path=None, with_optim=True, optim_path=None, optpara=None):
    if (optpara is None):
        optpara = {}
    model = mod(**args)
    optimizer, optimizer_sched = None, None
    if (with_optim):
        optimizer, optimizer_sched = get_default_optim(model.parameters(), **optpara)
    return model, optimizer, optimizer_sched


def get_default_logging_model(mod, args):
    model = mod(**args)
    return model, None, None
