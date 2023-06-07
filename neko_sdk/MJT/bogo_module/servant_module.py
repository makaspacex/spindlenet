class NekoStandBasic:
    def __init__(self, module):
        self.model = module

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
