from .field import RawField, Merge, ImageField, TextField
from .dataset import *
from torch.utils.data import DataLoader as TorchDataLoader

from .dataset import *


class DataLoader(TorchDataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, *args, collate_fn=dataset.collate_fn(), **kwargs)
