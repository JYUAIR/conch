from enum import Enum

from config import Config


class Dataset:
    def __init__(self, data, config: Config):
        self._data = data
        self._config = config
        self.train_set, self.validation_set, self.test_set = None, None, None

    def process_data(self):
        pass


class LoaderType(Enum):
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'


class DataLoader:
    def __init__(self, dataset: Dataset, config: Config):
        self._dataset = dataset
        self._config = config

    def __len__(self):
        pass

    def __iter__(self):
        pass
