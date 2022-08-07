import torch
from torch import nn, optim

from data.loading.TiSANCR_data import TiSANCR_DataLoader, TiSANCR_Dataset
from data.loading.base import Dataset, DataLoader, LoaderType
from experiment.net import TiSANCR
from config import Config
from config.base import Default


class Selector:
    """
        选择器
    """

    def __init__(self, config: Config):
        self.config = config

    def create_net(self, **kwargs) -> nn.Module:
        """
            创建网络模型
        :param kwargs: 用于放config中没有的参数
        """
        net_class = globals()[self.config.model]
        if net_class is TiSANCR:
            net = net_class(kwargs['n_users'], kwargs['n_items'], kwargs['n_timestamps'], self.config)

        return net

    def create_device(self) -> torch.device:
        return torch.device(self.config.device)

    def create_optimizer(self, **kwargs) -> torch.optim.Optimizer:
        if self.config.model == 'TiSANCR':
            optimizer = optim.Adam(kwargs['net_parameters'], lr=self.config.lr, weight_decay=self.config.l2)
        return optimizer

    def create_dataset(self, data, config: Config) -> Dataset:
        dataset_class = globals()[f'{self.config.model}_Dataset']
        if dataset_class is TiSANCR_Dataset:
            dataset = dataset_class(data, config)
        return dataset

    def create_dataloader(self, dataset: Dataset, dataloader_tyep: LoaderType) -> DataLoader:
        dataloader_class = globals()[f'{self.config.model}_DataLoader']
        _dataset = getattr(dataset, f'{dataloader_tyep.value}_set')
        if dataloader_class is TiSANCR_DataLoader:
            dataloader = dataloader_class(_dataset, self.config, dataloader_tyep, dataset._user_item_matrix)
        return dataloader


if __name__ == '__main__':
    c = Config(Default())
    setattr(c, 'model', 'TiSANCR')
    s = Selector(c)
    s.create_net()
