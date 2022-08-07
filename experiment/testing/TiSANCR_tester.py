from torch import nn
import numpy as np

from config import Config
from data.loading.base import Dataset
from experiment.monitor import MessageType
from experiment.testing.tester import Tester
from experiment.util.TiSANCR_util import logic_evaluate


class TiSANCR_Tester(Tester):
    def __init__(self, net: nn.Module, dataset: Dataset, config: Config, predict_function):
        super(TiSANCR_Tester, self).__init__(net, dataset, config)
        self.predict = predict_function

    def test(self):
        metric_dict = {}
        for i in range(self.config.n_times):
            evaluation_dict = logic_evaluate(self, self._test_dataloader, self.config.test_metrics)
            for metric in evaluation_dict:
                if metric not in metric_dict:
                    metric_dict[metric] = {}
                metric_mean = np.mean(evaluation_dict[metric])
                metric_std_err_val = np.std(evaluation_dict[metric]) / np.sqrt(len(evaluation_dict[metric]))
                if "mean" not in metric_dict[metric]:
                    metric_dict[metric]["mean"] = metric_mean
                    metric_dict[metric]["std"] = metric_std_err_val
                else:
                    metric_dict[metric]["mean"] += metric_mean
                    metric_dict[metric]["std"] += metric_std_err_val

        for metric in metric_dict:
            self.monitor.print_log({
                MessageType.TEXT: f'In testing, {metric}: {metric_dict[metric]["mean"] / self.config.n_times} ({metric_dict[metric]["std"] / self.config.n_times})'
            })