class NekoVlwrapper:
    def __init__(self, dataset):
        self.dataset = dataset
        self.it = iter(dataset)

    def get_batch(self):
        try:
            data = self.it.next()
        except StopIteration:
            # StopIteration is thrown if dataset_related ends
            # reinitialize data loader
            self.it = iter(self.dataset)
            data = next(self.it)
        return data
