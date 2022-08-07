import torch

import database
from config import Config


def load_checkpoint(id: int) -> (int, dict, dict, Config):
    db = database.ExperimentDatabase()
    sql = f"SELECT epoch, config FROM checkpoint WHERE id = {id}"
    rows = db.search(sql)
    rows = list(rows)
    if len(rows) == 0:
        return None
    checkpoint = torch.load(f'checkpoints/{id}.pt', map_location=torch.device('cpu'))
    net_state_dict = checkpoint['state_dict']
    optimizer_state_dict = checkpoint['optimizer']
    epoch, config = rows[0]
    config = Config.str2config(config)
    return epoch, net_state_dict, optimizer_state_dict, config


if __name__ == '__main__':
    print(load_checkpoint(2))
