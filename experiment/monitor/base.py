from enum import Enum


class EventListener:
    """
        监听器父类
    """

    def update(self, data):
        pass


class EventType(Enum):
    """
        事件类型枚举类
    """
    LOGGING = 'logging'
    CHECKPOINT = 'checkpoint'
    DRAWING = 'drawing'


class EventManager:
    def __init__(self):
        self._listeners: dict[EventType, EventListener] = {}

    def subscribe(self, event_type: EventType, listener: EventListener):
        """
            Listener订阅
        """
        self._listeners[event_type] = listener

    def unsubscribe(self, event_type: EventType):
        """
            取消订阅
        """
        self._listeners.pop(event_type, None)

    def notify(self, event_type: EventType, data):
        """
            更新提醒
        """
        self._listeners[event_type].update(data)


class Editor:
    """
        监视器更新类
    """
    def __init__(self):
        self.events = EventManager()

    def print_log(self, msg: dict):
        self.events.notify(EventType.LOGGING, msg)

    def save_checkpoint(self, data: dict):
        self.events.notify(EventType.CHECKPOINT, data)

    def draw(self, data: tuple):
        self.events.notify(EventType.DRAWING, data)
