from experiment.monitor.base import EventListener
from database import ExperimentDatabase
from enum import Enum


class MessageType(Enum):
    """
        信息类型
        TEXT 用于放难以归类的信息，建议在log末尾
    """
    EPOCH = 'epoch'
    BATCH = 'batch'
    TOTAL_BATCHES = 'total batches'
    LOSS = 'loss'
    TIME = 'time'
    TEXT = 'text'


class LoggingListener(EventListener):
    def __init__(self):
        self.db = ExperimentDatabase()
        self.__new_logger()
        self.__log_file = open(f'log/{self.id}.log', 'w')

    def __del__(self):
        self.__log_file.close()

    def __new_logger(self):
        sql = '''INSERT INTO logging (timestamp) 
        VALUES (datetime(current_timestamp, 'localtime'))
        '''
        self.db.insert(sql)
        self.id = self.db.get_last_id()

    def update(self, data):
        msg = self.__message_creator(data)
        print(msg)
        self.__log_file.write(f'{msg}\n')
        self.__log_file.flush()

    def __message_creator(self, data) -> str:
        ret = '| '
        for msg_type in list(MessageType):
            try:
                ret += f'{msg_type.value}: {data[msg_type]} |'
            except KeyError:
                pass
        return ret

    def _init_database(self):
        """
            初始化数据库
        """
        sql = '''create table logging
      (
          id     integer not null
              constraint logging_pk
                  primary key autoincrement,
          timestamp text not null
      )'''
        self.db.create_table(sql)


if __name__ == '__main__':
    d = {
        MessageType.EPOCH: "test"
    }
    logger = LoggingListener()
    # logger.update(d)
    # logger._initDatabase()
    print(logger.id)
