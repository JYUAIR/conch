import sqlite3

# db_file = 'experiment_data.sqlite'
db_file = '/Users/h/Project/conch/experiment_data.sqlite'


class ExperimentDatabase:
    def __init__(self):
        self.__db_connector = sqlite3.connect(db_file)
        self.__db_cursor = self.__db_connector.cursor()

    def __del__(self):
        self.__db_connector.close()

    def get_last_id(self):
        return self.__db_cursor.lastrowid

    def insert(self, sql: str):
        self.__db_cursor.execute(sql)
        self.__db_connector.commit()

    def create_table(self, sql: str):
        self.__db_cursor.execute(sql)

    def search(self, sql: str):
        return self.__db_cursor.execute(sql)
