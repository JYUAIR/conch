import json

import toml

import argparse


class Default:
    """
        默认配置基础类
        须在defaultConfiguration方法中
        声明所有参数以及其类型和默认值
        没有默认值的，使用None
    """

    def __init__(self):
        self._config_pairs = {}

    def __default_configuration(self):
        pass

    def add_config_pair(self, key: str, value):
        self._config_pairs[key] = value

    def set_config_pair(self, key: str, value):
        if key not in self._config_pairs.keys():
            raise Exception(f'Try to set parameter {key}, but {key} is not in the preset')
        if type(value) != type(self._config_pairs[key]):
            raise Exception(
                f'Try to set parameter {key}, but the required type is {type(self._config_pairs[key])} instead of {type(value)}')
        self._config_pairs[key] = value

    def get_config_pairs(self) -> dict:
        return self._config_pairs

    def get_config_pair(self, key: str):
        return self._config_pairs[key]

    def __str__(self):
        return str(json.dumps(self._config_pairs))


class Config(object):
    """
        配置优先级顺序：1 命令行参数
                     2 配置文件
                     3 默认类
    """

    def __init__(self, default: Default, path: str = None):
        self._setting = default
        if path is not None:
            config = self.__load_file(path)
            self.__load(config)
        config = self.__load_argv()
        self.__load(config)

    def __getattribute__(self, name: str):
        if name == '_setting' or name not in self._setting.get_config_pairs().keys():
            return object.__getattribute__(self, name)
        return self._setting.get_config_pair(name)

    def config2str(self):
        return str(self._setting.get_config_pairs())

    @classmethod
    def str2config(cls, config_str: str):
        config = Config(Default())
        config._setting._config_pairs = json.loads(config_str)
        return config

    def __load(self, config):
        for key, value in config.items():
            self._setting.set_config_pair(key, value)

    def __load_file(self, path: str):
        return toml.load(path)

    def __load_argv(self):
        parser = argparse.ArgumentParser()
        for pair in self._setting.get_config_pairs().items():
            parser.add_argument(f'--{pair[0]}', type=type(pair[1]), default=pair[1])
        args, _ = parser.parse_known_args()
        return vars(args)

    def __str__(self):
        return str(self._setting)


class testdefault(Default):
    """
        参考类型
        可参照此类自定义自己的默认值类型
    """

    def __init__(self):
        super(testdefault, self).__init__()
        self.__default_configuration()

    def __default_configuration(self):
        self.add_config_pair('test', 123)


if __name__ == '__main__':
    d = testdefault()
    d.set_config_pair('test', 456)
    c = Config(d)
    print(c.test)
