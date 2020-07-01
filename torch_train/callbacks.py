import torch
from torch import nn

class CallBack(object):
    """Base class of callbacks
    """

    def __init__(self):
        pass

    def step(self):
        pass


class EarlyStoping(CallBack):
    """Stop training when metric on validation set not imporved
    Args:
        mode: str, 'min' or 'max', metric used to define 'improved'
        delta: float, if new metric is in delta range of old metric, no improved
        patience: int, times to wait for improved
    """

    def __init__(self, model, mode='max', delta=0, patience=5, path='./checkpoint/last_best.pt'):
        super(EarlyStoping, self).__init__()
        self.mode = mode
        self.delta = delta
        self.model = model
        self.path = path
        self.ori_patience = patience
        self.cur_patience = patience

        self.last_best = None
        self.sign = 1 if mode == 'max' else -1

    def step(self, value):
        # when first call step
        if self.last_best is None:
            self.last_best = value
            return

        if self.sign * (value - self.last_best) > self.delta:
            # better than before
            self.cur_patience = self.ori_patience
            torch.save(self.model.state_dict(),self.path)
            self.last_best = value
        else:
            self.cur_patience -= 1
            if self.cur_patience <= 0:
                raise StopIteration


class LinearDecay(CallBack):
    pass