import pandas as pd
import numpy as np
import random
import torch

from config import Config
from config.TiSANCR_default import TiSANCR_Default
from data.loading.TiSANCR_data import TiSANCR_Dataset
from experiment.training.TiSANCR_trainer import TiSANCR_Trainer
from experiment.testing.TiSANCR_tester import TiSANCR_Tester


def main():
    config = Config(TiSANCR_Default())

    # 为了实验可复现，设置随机种子
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    raw_data = pd.read_csv(f'datasets/{config.dataset}/data.csv')
    dataset = TiSANCR_Dataset(raw_data, config)

    trainer = TiSANCR_Trainer(dataset, config)
    model_id = trainer.train()
    tester = TiSANCR_Tester(trainer._net, dataset, config, trainer.predict)
    tester.load_checkpoint(model_id)
    tester.test()


if __name__ == '__main__':
    main()
