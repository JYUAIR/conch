import copy

from pandas import DataFrame
import pandas as pd
from scipy import sparse

from config import Config
from data.processing.base import Strategy


class UitrNStrategy(Strategy):
    """
        csv 推荐系统数据
        类型初始化、数据量计算
    """

    def execute(self, data: DataFrame, config: Config = None):
        data = data.fillna(0)
        data['timestamp'] = data['timestamp'].astype(int)
        data['rating'] = data['rating'].astype(int)
        n_users = data['userID'].nunique()
        n_items = data['itemID'].nunique()
        n_timestamps = data['timestamp'].max() // 86400 + 1
        return data, n_users, n_items, n_timestamps


class UiMatrixStrategy(Strategy):
    def execute(self, data, config: Config = None):
        data, n_users, n_items, n_timestamps = data
        group = data.groupby('userID')
        rows, cols = [], []
        values = []
        for i, (_, g) in enumerate(group):
            u = list(g['userID'])[0]
            items = set(list(g['itemID']))
            rows.extend([u] * len(items))
            cols.extend(list(items))
            values.extend([1] * len(items))
        return data, n_users, n_items, n_timestamps, \
               sparse.csr_matrix((values, (rows, cols)), (n_users, n_items))


class TiSANCR_DivisionStrategy(Strategy):
    def execute(self, data: DataFrame, config: Config = None):
        proc_data = data.copy()
        proc_data['rating'][proc_data['rating'] < config.threshold] = 0
        proc_data['rating'][proc_data['rating'] >= config.threshold] = 1
        if config.order:
            proc_data = proc_data.sort_values(by=['timestamp', 'userID', 'itemID']).reset_index(
                drop=True)
        return proc_data


class TiSANCR_LeaveOneOutStrategy(Strategy):
    def execute(self, data: DataFrame, config: Config = None):
        train_set = []
        processed_data = data.copy()
        for uid, group in processed_data.groupby('userID'):
            found, found_idx = 0, -1
            for idx in group.index:
                if group.loc[idx, 'rating'] > 0:
                    found_idx = idx
                    found += 1
                    if found >= config.keep_n:
                        break
            if found_idx > 0:
                train_set.append(group.loc[:found_idx])
        train_set = pd.concat(train_set)
        processed_data = processed_data.drop(train_set.index)

        test_set = []
        for uid, group in processed_data.groupby('userID'):
            found, found_idx = 0, -1
            for idx in reversed(group.index):
                if group.loc[idx, 'rating'] > 0:
                    found_idx = idx
                    found += 1
                    if found >= config.leave_n:
                        break
            if found_idx > 0:
                test_set.append(group.loc[found_idx:])
        test_set = pd.concat(test_set)
        processed_data = processed_data.drop(test_set.index)

        validation_set = []
        for uid, group in processed_data.groupby('userID'):
            found, found_idx = 0, -1
            for idx in reversed(group.index):
                if group.loc[idx, 'rating'] > 0:
                    found_idx = idx
                    found += 1
                    if found >= config.leave_n:
                        break
            if found_idx > 0:
                validation_set.append(group.loc[found_idx:])
        validation_set = pd.concat(validation_set)
        processed_data = processed_data.drop(validation_set.index)

        train_set = pd.concat([train_set, processed_data])
        validation_set, test_set = validation_set.reset_index(drop=True), test_set.reset_index(drop=True)
        return train_set, validation_set, test_set


class TiSANCR_GenerateHistoriesStrategy(Strategy):
    def execute(self, data, config: Config = None):
        train_set, validation_set, test_set = data
        history_dict = {}
        feedback_dict = {}
        his_time_dict = {}
        for df in [train_set, validation_set, test_set]:
            history = []
            fb = []

            hist_len = []
            his_time = []
            uids, iids, feedbacks, timestamps = df['userID'].tolist(), df['itemID'].tolist(), df['rating'].tolist(), \
                                                df['timestamp'].tolist()
            for i, uid in enumerate(uids):
                iid, feedback, timestamp = iids[i], feedbacks[i], timestamps[i]

                if uid not in history_dict:
                    history_dict[uid] = []
                    feedback_dict[uid] = []
                    his_time_dict[uid] = []

                tmp_his = copy.deepcopy(history_dict[uid]) if config.max_history_length == 0 else history_dict[uid][
                                                                                               -config.max_history_length:]

                fb_his = copy.deepcopy(feedback_dict[uid]) if config.max_history_length == 0 else feedback_dict[uid][
                                                                                               -config.max_history_length:]

                tmp_his_time = copy.deepcopy(his_time_dict[uid]) if config.max_history_length == 0 else his_time_dict[uid][
                                                                                                     -config.max_history_length:]

                history.append(tmp_his)
                fb.append(fb_his)
                hist_len.append(len(tmp_his))
                his_time.append(tmp_his_time)

                history_dict[uid].append(iid)
                feedback_dict[uid].append(feedback)
                his_time_dict[uid].append(timestamp)

            df['history'] = history
            df['history_feedback'] = fb
            df['history_length'] = hist_len
            df['history_timestamp'] = his_time

        if config.premise_threshold != 0:
            train_set = train_set[train_set.history_length > config.premise_threshold]
            validation_set = validation_set[validation_set.history_length > config.premise_threshold]
            test_set = test_set[test_set.history_length > config.premise_threshold]
        return train_set, validation_set, test_set

class TiSANCR_ResetIndexStrategy(Strategy):
    def execute(self, data, config: Config = None):
        train_set, validation_set, test_set = data
        train_set = train_set[train_set['rating'] > 0].reset_index(drop=True)
        train_set = train_set[train_set['history_feedback'].map(len) > 0].reset_index(drop=True)
        validation_set = validation_set[validation_set['rating'] > 0].reset_index(drop=True)
        validation_set = validation_set[validation_set['history_feedback'].map(len) > 0].reset_index(
            drop=True)
        test_set = test_set[test_set['rating'] > 0].reset_index(drop=True)
        test_set = test_set[test_set['history_feedback'].map(len) > 0].reset_index(drop=True)
        return train_set, validation_set, test_set
