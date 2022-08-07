import torch

from experiment.monitor.base import EventListener
from database import ExperimentDatabase


class CheckpointListener(EventListener):
    def __init__(self, save_interval: int = 0):
        self.save_interval = save_interval
        self.db = ExperimentDatabase()

    def update(self, data):
        epoch, net_state_dict, optimizer_state_dict, log_id, config = data
        if self.save_interval == 0 or epoch % self.save_interval == 0:
            sql = f'''INSERT INTO checkpoint (epoch, log_id, config)
                      VALUES ({epoch}, {log_id}, '{str(config)}')'''
            self.db.insert(sql)
            self.model_id = self.db.get_last_id()
            torch.save({
                'state_dict': net_state_dict,
                'optimizer': optimizer_state_dict
            }, f'checkpoints/{self.model_id}.pt')

    def _init_database(self):
        """
            初始化数据库
        """
        sql = '''create table checkpoint
      (
          id     integer not null
              constraint checkpoint_pk
                  primary key autoincrement,
          epoch  integer not null,
          log_id integer not null,
          config text
      )'''
        self.db.create_table(sql)


if __name__ == '__main__':
    c = CheckpointListener()
    # c.update((10, None, None, '{"test": 123}'))
    # c.update((10, None, None, '{"test": 123}'))
    c._init_database()
    """
    INSERT INTO checkpoint (epoch, log_id, config)
                      VALUES (1, 27, json('{"verbose": -1, "threshold": 4, "order": true, "leave_n": 1, "keep_n": 5, "max_history_length": 5, "n_neg_train": 1, "n_neg_val_test": 100, "training_batch_size": 128, "val_test_batch_size": 256, "seed": 2021, "emb_size": 64, "dropout": 0.0, "lr": 0.001, "l2": 0.0001, "r_weight": 0.01, "val_metric": "ndcg@5", "test_metrics": ["ndcg@5", "ndcg@10", "hit@5", "hit@10"], "n_epochs": 100, "early_stop": 0, "at_least": 20, "n_times": 10, "dataset": "movielens_100k", "premise_threshold": 0, "device": "cpu", "model": "TiSANCR", "use_multi_gpu": false, "attention_heads": 8}'))
    """
