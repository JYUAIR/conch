from config.base import Default


class TiSANCR_Default(Default):
    def __init__(self):
        super(TiSANCR_Default, self).__init__()
        self.__default_configuration()

    def __default_configuration(self):
        self.add_config_pair('verbose', -1)
        self.add_config_pair('threshold', 4)
        self.add_config_pair('order', True)
        self.add_config_pair('leave_n', 1)
        self.add_config_pair('keep_n', 5)
        self.add_config_pair('max_history_length', 5)
        self.add_config_pair('n_neg_train', 1)
        self.add_config_pair('n_neg_val_test', 100)
        self.add_config_pair('training_batch_size', 128)
        self.add_config_pair('val_test_batch_size', 256)
        self.add_config_pair('seed', 2021)
        self.add_config_pair('emb_size', 64)
        self.add_config_pair('dropout', 0.0)
        self.add_config_pair('lr', 0.001)
        self.add_config_pair('l2', 0.0001)
        self.add_config_pair('r_weight', 0.01)
        self.add_config_pair('val_metric', 'ndcg@5')
        self.add_config_pair('test_metrics', ['ndcg@5', 'ndcg@10', 'hit@5', 'hit@10'])
        self.add_config_pair('n_epochs', 100)
        self.add_config_pair('early_stop', 0)
        self.add_config_pair('at_least', 20)
        self.add_config_pair('n_times', 10)
        self.add_config_pair('dataset', 'movielens_100k')
        self.add_config_pair('premise_threshold', 0)
        self.add_config_pair('device', 'cpu')
        self.add_config_pair('model', 'TiSANCR')
        self.add_config_pair('use_multi_gpu', False)
        self.add_config_pair('attention_heads', 8)
