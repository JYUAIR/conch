import matlab
from matlab import engine
import numpy as np

import random
from random import choice

from experiment.monitor.base import EventListener


class DrawingListener(EventListener):
    def draw(self, img_name, x, x_label, x_limit, ys, y_label, y_limit, line_style, legend):
        eng = engine.start_matlab()
        eng.figure(nargout=0)
        x = matlab.double(x)
        for y, style in zip(ys, line_style):
            eng.plot(x, matlab.double(y), style, 'lineWidth', 1.5)
            eng.hold('on', nargout=0)
        eng.xlim(matlab.double(x_limit), nargout=0)
        eng.ylim(matlab.double(y_limit), nargout=0)
        eng.xlabel(x_label)
        eng.ylabel(y_label)
        eng.legend(*legend)
        eng.savefig(f'img/{img_name}.fig', nargout=0)
        eng.close()

    matlab_line_style = ['-', '--', ':', '-.']
    matlab_marker = ['o', '+', '*', '.', 'x', '_', '|', 's', 'd', '^', 'v', '>', '<', 'p', 'h']
    matlab_color = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    def random_line_style(self, line_style: list, length: int) -> list:
        if len(line_style) == length:
            return line_style
        style = f'{choice(DrawingListener.matlab_line_style)}{choice(DrawingListener.matlab_color)}{choice(DrawingListener.matlab_marker)}'
        if style not in line_style:
            line_style.append(style)
        return self.random_line_style(line_style, length)

    def create_section(self, nums) -> list:
        section = [np.min(nums), np.max(nums)]
        mid = (section[0] + section[1]) / 2
        section[0], section[1] = section[0] - mid, section[1] + mid
        return section

    def update(self, data):
        img_name, x, x_label, x_limit, ys, y_label, y_limit, legend = data
        if x_limit is None:
            x_limit = self.create_section(x)
        if y_limit is None:
            y_limit = self.create_section(ys)
        if x_label is None:
            x_label = 'x'
        if y_label is None:
            y_label = 'y'
        if legend is None:
            legend = [f'line{i}' for i in range(1, len(ys) + 1)]
        line_style = self.random_line_style([], len(ys))
        self.draw(img_name, x, x_label, x_limit, ys, y_label, y_limit, line_style, legend)
