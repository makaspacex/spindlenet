class neko_res45_binorm_bogo:
    def cuda(self):
        pass

    def freeze(self):
        self.container.eval()

    def freezebn(self):
        self.container.model.freezebnprefix(self.bnname)

    def unfreezebn(self):
        self.container.model.unfreezebnprefix(self.bnname)

    def unfreeze(self):
        self.container.model.train()

    def get_torch_module_dict(self):
        return self.container.model

    def __init__(self, args, mod_dict):
        self.container = mod_dict[args["container"]]
        self.name = args["name"]
        self.bnname = args["name"].replace("res", "bn")
        self.model = mod_dict[args["container"]].model.bogo_modules[args["name"]]

    def __call__(self, x):
        return self.model(x)


class neko_res45cco_binorm_bogo:
    def cuda(self):
        pass

    def freeze(self):
        self.container.eval()

    def freezebn(self):
        self.container.model.freezebnprefix(self.bnname)

    def unfreezebn(self):
        self.container.model.unfreezebnprefix(self.bnname)

    def unfreeze(self):
        self.container.model.train()

    def get_torch_module_dict(self):
        return self.container.model

    def __init__(self, args, mod_dict):
        self.container = mod_dict[args["container"]]
        self.name = args["name"]
        self.bnname = args["name"].replace("res", "bn")
        self.model = mod_dict[args["container"]].model.bogo_modules[args["name"]]

    def __call__(self, x):
        return self.model(x)
