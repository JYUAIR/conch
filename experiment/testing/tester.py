from torch import nn

from data.loading.base import Dataset, LoaderType
from config.base import Config
from experiment.monitor import Editor, MessageType, EventType
from experiment.monitor.drawing import DrawingListener
from experiment.monitor.logger import LoggingListener
from experiment.util import load_checkpoint
from experiment.util.selection import Selector


class Tester:
    def __init__(self, net: nn.Module, dataset: Dataset, config: Config):
        self.config = config
        self._net = net
        self._selector = Selector(self.config)
        self._test_dataloader = self._selector.create_dataloader(dataset, LoaderType.TEST)
        self.device = self._selector.create_device()
        self.monitor = Editor()
        self.__init_monitor()

    def __init_monitor(self):
        self.monitor.events.subscribe(EventType.LOGGING, LoggingListener())
        self.monitor.events.subscribe(EventType.DRAWING, DrawingListener())

    def load_checkpoint(self, model_id: int):
        self.monitor.print_log({MessageType.TEXT: f'Loading checkpoint id {model_id}'})
        _, net_state_dict, optimizer_state_dict, _ = load_checkpoint(model_id)
        self._net.load_state_dict(net_state_dict)
        self.monitor.print_log({MessageType.TEXT: 'Checkpoint loaded'})

    def test(self):
        pass
