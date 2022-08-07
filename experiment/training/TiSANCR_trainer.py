import torch
import torch.nn.functional as F
import numpy as np

import time

from config import Config
from data.loading.TiSANCR_data import TiSANCR_Dataset
from data.loading.base import Dataset
from experiment.monitor import MessageType, EventType
from experiment.training import Trainer
from experiment.util.TiSANCR_util import ValidFunc, logic_evaluate


class TiSANCR_Trainer(Trainer):
    def __init__(self, dataset: TiSANCR_Dataset, config: Config):
        super(TiSANCR_Trainer, self).__init__(dataset, config)
        self.create_net()
        self.create_optimizer()

    def create_net(self):
        self._net = self._selector.create_net(n_users=self._dataset.n_users, n_items=self._dataset.n_items,
                                              n_timestamps=self._dataset.n_timestamps)

    def create_optimizer(self):
        self._optimizer = self._selector.create_optimizer(net_parameters=self._net.parameters())

    def reg_loss(self, constraints):
        network = self._net
        if self.config.use_multi_gpu:
            network = self._net.module
        false_vector = network.logic_not(network.true_vector)

        r_not_not_true = (1 - F.cosine_similarity(
            network.logic_not(network.logic_not(network.true_vector)), network.true_vector, dim=0))

        r_not_not_self = \
            (1 - F.cosine_similarity(network.logic_not(network.logic_not(constraints)), constraints)).mean()

        r_not_self = (1 + F.cosine_similarity(network.logic_not(constraints), constraints)).mean()

        r_not_not_not = \
            (1 + F.cosine_similarity(network.logic_not(network.logic_not(constraints)),
                                     network.logic_not(constraints))).mean()

        r_or_true = (1 - F.cosine_similarity(
            network.logic_or(constraints, network.true_vector.expand_as(constraints)),
            network.true_vector.expand_as(constraints))).mean()

        r_or_false = (1 - F.cosine_similarity(
            network.logic_or(constraints, false_vector.expand_as(constraints)), constraints)).mean()

        r_or_self = (1 - F.cosine_similarity(network.logic_or(constraints, constraints), constraints)).mean()

        r_or_not_self = (1 - F.cosine_similarity(
            network.logic_or(constraints, network.logic_not(constraints)),
            network.true_vector.expand_as(constraints))).mean()

        r_or_not_self_inverse = (1 - F.cosine_similarity(
            network.logic_or(network.logic_not(constraints), constraints),
            network.true_vector.expand_as(constraints))).mean()

        r_and_true = (1 - F.cosine_similarity(
            network.logic_and(constraints, network.true_vector.expand_as(constraints)), constraints)).mean()

        r_and_false = (1 - F.cosine_similarity(
            network.logic_and(constraints, false_vector.expand_as(constraints)),
            false_vector.expand_as(constraints))).mean()

        r_and_self = (1 - F.cosine_similarity(network.logic_and(constraints, constraints), constraints)).mean()

        r_and_not_self = (1 - F.cosine_similarity(
            network.logic_and(constraints, network.logic_not(constraints)),
            false_vector.expand_as(constraints))).mean()

        r_and_not_self_inverse = (1 - F.cosine_similarity(
            network.logic_and(network.logic_not(constraints), constraints),
            false_vector.expand_as(constraints))).mean()

        true_false = 1 + F.cosine_similarity(network.true_vector, false_vector, dim=0)

        r_loss = r_not_not_true + r_not_not_self + r_not_self + r_not_not_not + \
                 r_or_true + r_or_false + r_or_self + r_or_not_self + r_or_not_self_inverse + true_false + \
                 r_and_true + r_and_false + r_and_self + r_and_not_self + r_and_not_self_inverse

        return r_loss

    def loss_function(self, positive_preds, negative_preds, constraints):
        positive_preds = positive_preds.view(positive_preds.size(0), 1)
        positive_preds = positive_preds.expand(positive_preds.size(0), negative_preds.size(1))
        loss = -(positive_preds - negative_preds).sigmoid().log().sum()

        r_loss = self.reg_loss(constraints)

        return loss + self.config.r_weight * r_loss

    def predict(self, batch_data):
        self._net.eval()
        with torch.no_grad():
            positive_predictions, negative_predictions, _ = self._net(batch_data)
        return positive_predictions, negative_predictions

    def train(self):
        valid_func = ValidFunc(logic_evaluate)
        best_val = 0.0
        early_stop_counter = 0
        early_stop_flag = False
        if self.config.early_stop > 1:
            early_stop_flag = True
        try:
            mu_vals = []
            for epoch in range(1, self.config.n_epochs + 1):
                self.train_epoch(epoch)
                if self._vali_dataloader is not None:
                    valid_res = valid_func(self, self._vali_dataloader, self.config.val_metric)
                    mu_val = np.mean(valid_res)
                    mu_vals.append(mu_val)
                    self.monitor.draw((f'{self.monitor.events._listeners[EventType.LOGGING].id}',
                                       [i for i in range(1, epoch + 1)], 'epoch', None, [mu_vals],
                                       self.config.val_metric.upper(), None, None))
                    if mu_val > best_val:
                        best_val = mu_val
                        self.epoch = epoch
                        self.monitor.print_log({
                            MessageType.TEXT: f'{self.config.val_metric}: {mu_val}'
                        })
                        self.save_checkpoint()
                        early_stop_counter = 0
                    else:
                        if epoch >= self.config.at_least and early_stop_flag:
                            early_stop_counter += 1
                            if early_stop_counter == self.config.early_stop:
                                self.monitor.print_log({
                                    MessageType.EPOCH: epoch,
                                    MessageType.TEXT: 'Traing stopped due to early stopping'
                                })
                                break
        except KeyboardInterrupt:
            self.monitor.print_log({MessageType.TEXT: 'Handled KeyboardInterrupt: exiting from training early'})
        return self.monitor.events._listeners[EventType.CHECKPOINT].model_id
    def train_epoch(self, epoch):
        self._net.train()
        train_loss = 0
        partial_loss = 0
        epoch_start_time = time.time()
        log_delay = max(10, len(self._train_dataloader) // 10 ** self.config.verbose)

        for batch_idx, batch_data in enumerate(self._train_dataloader):
            partial_loss += self.train_batch(batch_data)
            if (batch_idx + 1) % log_delay == 0:
                self.monitor.print_log({
                    MessageType.EPOCH: epoch,
                    MessageType.BATCH: (batch_idx + 1),
                    MessageType.TOTAL_BATCHES: len(self._train_dataloader),
                    MessageType.LOSS: partial_loss / log_delay
                })
                train_loss += partial_loss
                partial_loss = 0.0
        total_loss = (train_loss + partial_loss) / len(self._train_dataloader)
        time_diff = time.time() - epoch_start_time
        self.monitor.print_log({
            MessageType.EPOCH: epoch,
            MessageType.LOSS: total_loss,
            MessageType.TIME: time_diff
        })

    def train_batch(self, batch_data):
        self._optimizer.zero_grad()
        positive_preds, negative_preds, constraints = self._net(batch_data)
        loss = self.loss_function(positive_preds, negative_preds, constraints)
        loss.backward()
        self._optimizer.step()
        return loss.item()
