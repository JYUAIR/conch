from config import Config
from data.loading.base import Dataset, LoaderType
from experiment.monitor import Editor, EventType
from experiment.monitor.checkpoint import CheckpointListener
from experiment.monitor.drawing import DrawingListener
from experiment.monitor.logger import LoggingListener, MessageType
from experiment.util.selection import Selector
from experiment.util import load_checkpoint


class Trainer:
    def __init__(self, dataset: Dataset, config: Config):
        self.config = config
        self._selector = Selector(self.config)
        self._dataset = dataset
        self._train_dataloader = self._selector.create_dataloader(dataset, LoaderType.TRAIN)
        self._vali_dataloader = self._selector.create_dataloader(dataset, LoaderType.VALIDATION)
        self.device = self._selector.create_device()
        self.monitor = Editor()
        self.__init_monitor()
        self.epoch = 1
        self._net = None
        self._optimizer = None

    def create_net(self):
        """
            创建网络模型 self._net
        """
        pass

    def create_optimizer(self):
        pass

    def __init_monitor(self):
        self.monitor.events.subscribe(EventType.LOGGING, LoggingListener())
        self.monitor.events.subscribe(EventType.CHECKPOINT, CheckpointListener())
        self.monitor.events.subscribe(EventType.DRAWING, DrawingListener())

    def train(self):
        pass

    def save_checkpoint(self):
        self.monitor.print_log({MessageType.TEXT: 'Saving checkpoint'})
        checkpoint = (self.epoch, self._net.state_dict(), self._optimizer.state_dict(),
                      self.monitor.events._listeners[EventType.LOGGING].id, self.config)
        self.monitor.save_checkpoint(checkpoint)
        self.monitor.print_log({MessageType.TEXT: 'Checkpoint saved'})

    def load_checkpoint(self, model_id: int):
        self.monitor.print_log({MessageType.TEXT: f'Loading checkpoint id {model_id}'})
        self.epoch, net_state_dict, optimizer_state_dict, self.config = load_checkpoint(model_id)
        self._net.load_state_dict(net_state_dict)
        self._optimizer.load_state_dict(optimizer_state_dict)
        self.config = Config.str2config(self.config)
        self.monitor.print_log({MessageType.TEXT: 'Checkpoint loaded'})


if __name__ == '__main__':
    Trainer(None)
