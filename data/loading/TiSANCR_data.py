from pandas import DataFrame

import torch
import numpy as np
import random

from config import Config
from data import Context
from data.loading.base import Dataset, DataLoader, LoaderType
from data.processing.csv import UitrNStrategy, UiMatrixStrategy, TiSANCR_DivisionStrategy, TiSANCR_LeaveOneOutStrategy, \
    TiSANCR_GenerateHistoriesStrategy, TiSANCR_ResetIndexStrategy


class TiSANCR_Dataset(Dataset):
    def __init__(self, data: DataFrame, config: Config):
        super(TiSANCR_Dataset, self).__init__(data, config)
        self.process_data()

    def process_data(self):
        ctx1 = Context()
        ctx1.add_strategy(UitrNStrategy())
        ctx1.add_strategy(UiMatrixStrategy())
        self._data, self.n_users, self.n_items, self.n_timestamps, self._user_item_matrix \
            = ctx1.execute_strategies(self._data)

        ctx2 = Context()
        ctx2.add_strategy(TiSANCR_DivisionStrategy())
        ctx2.add_strategy(TiSANCR_LeaveOneOutStrategy())
        ctx2.add_strategy(TiSANCR_GenerateHistoriesStrategy())
        ctx2.add_strategy(TiSANCR_ResetIndexStrategy())
        self.train_set, self.validation_set, self.test_set = ctx2.execute_strategies(self._data, self._config)


class TiSANCR_DataLoader(DataLoader):
    def __init__(self, dataset: TiSANCR_Dataset, config: Config, loader_type: LoaderType, user_item_matrix):
        super(TiSANCR_DataLoader, self).__init__(dataset, config)
        self._device = torch.device(self._config.device)
        self._user_item_matrix = user_item_matrix

        if loader_type is LoaderType.TRAIN:
            self.batch_size = self._config.training_batch_size
            self.shuffle = True
            self.n_neg_samples = self._config.n_neg_train
        else:
            self.batch_size = self._config.val_test_batch_size
            self.shuffle = False
            self.n_neg_samples = self._config.n_neg_val_test


    def __len__(self):
        dataset_size = 0
        length = self._dataset.groupby("history_length")
        for i, (_, l) in enumerate(length):
            dataset_size += int(np.ceil(l.shape[0] / self.batch_size))
        return dataset_size

    def __iter__(self):
        length = self._dataset.groupby('history_length')

        for i, (_, l) in enumerate(length):
            group_users = np.array(list(l['userID']))
            group_items = np.array(list(l['itemID']))
            group_histories = np.array(list(l['history']))
            group_feedbacks = np.array(list(l['history_feedback']))

            group_timestamp = np.array(list(l['timestamp'])) // 86400
            group_history_timestamp = np.array(list(l['history_timestamp'])) // 86400

            n = group_users.shape[0]
            idxlist = list(range(n))
            if self.shuffle:
                np.random.shuffle(idxlist)

            for _, start_idx in enumerate(range(0, n, self.batch_size)):
                end_idx = min(start_idx + self.batch_size, n)
                batch_users = torch.from_numpy(group_users[idxlist[start_idx:end_idx]])
                batch_items = torch.from_numpy(group_items[idxlist[start_idx:end_idx]])
                batch_histories = torch.from_numpy(group_histories[idxlist[start_idx:end_idx]])
                batch_feedbacks = torch.from_numpy(group_feedbacks[idxlist[start_idx:end_idx]])
                batch_timestamp = torch.from_numpy(group_timestamp[idxlist[start_idx:end_idx]])
                batch_history_timestamp = torch.from_numpy(group_history_timestamp[idxlist[start_idx:end_idx]])

                batch_user_item_matrix = self._user_item_matrix[batch_users].toarray()

                batch_user_unseen_items = 1 - batch_user_item_matrix

                negative_items = []
                for u in range(batch_users.size(0)):
                    u_unseen_items = batch_user_unseen_items[u].nonzero()[0]

                    rnd_negatives = u_unseen_items[
                        random.sample(range(u_unseen_items.shape[0]), self.n_neg_samples)]

                    negative_items.append(rnd_negatives)
                batch_negative_items = torch.tensor(negative_items)

                if not self._config.use_multi_gpu:
                    yield batch_users.to(self._device), batch_items.to(self._device), batch_histories.to(self._device), \
                          batch_feedbacks.to(self._device), batch_negative_items.to(self._device), \
                          batch_timestamp.to(self._device), batch_history_timestamp.to(self._device)
                else:
                    yield batch_users, batch_items, batch_histories, \
                          batch_feedbacks, batch_negative_items, \
                          batch_timestamp, batch_history_timestamp
