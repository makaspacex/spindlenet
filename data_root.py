
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

def find_data_root():
    return "/home/izhangxm/mnt/hotssd/recdatassd/"


def find_model_root():
    return "/home/izhangxm/mnt/hotssd/models-release"


def find_export_root():
    return "/home/izhangxm/mnt/hotssd/results"
