from config import Config


class Strategy:
    """
        预处理父类
        所有子类必须实现execute方法
    """

    def execute(self, data, config: Config = None):
        pass


class Context:
    """
        预处理策略上下类
        添加所有所需策略
        顺序执行策略
    """

    def __init__(self):
        self.__strategies: [Strategy] = []

    def add_strategy(self, strategy: Strategy):
        self.__strategies.append(strategy)

    def execute_strategies(self, data, config: Config = None):
        for strategy in self.__strategies:
            data = strategy.execute(data, config)
        return data
