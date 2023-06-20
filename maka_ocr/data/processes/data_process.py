class DataProcess(object):
    r"""Processes of data dict."""

    def __call__(self, data: dict):
        raise NotImplementedError


class Compose(DataProcess):
    def __init__(self, processes: list[DataProcess]) -> None:
        self.processes = processes

    def __call__(self, data:dict):
        for process in self.processes:
            data = process(data)
        return data
