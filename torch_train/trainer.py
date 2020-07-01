import torch
from torch import nn
import numpy as np
import random
from tqdm import tqdm
from collections import deque

# TODO: distributed(): set to multi-gpu
# TODO: add logger


class Trainer(object):
    def __init__(self, device=None, verbose=True, name='trainer'):
        self.device = device if device else \
            torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.name = name
        self.verbose = verbose
        self.setup = False

    def build(self, model, optimizer, criterion, callbacks=None, metric=None):
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = callbacks
        self.metric = metric
        self.setup = True

    def fit(self, train_loader, num_epoch, validate_loader=None):
        # check everything
        # TODO: add check module or function to replace this
        assert self.setup == True, 'You should build trainer before fit'
        if any([validate_loader, self.metric]):
            assert all([validate_loader, self.metric])
        # set verbose using tqdm
        if verbose:
            train_loader = tqdm(train_loader)
            validate_loader = tqdm(validate_loader) if validate_loader else None
        # train loop
        for epoch in range(num_epoch):
            loss = []
            for i, data in enumerate(train_loader):
                # TODO: use Summarizer to record info, not print
                train_info = self._train_step()
                loss.append(train_info['loss'])
            print('epoch {}, current train loss: {}'.format(i+1, np.mean(loss)))
            # validate loop
            if validate_loader:
                # TODO: find a more elegant way to pack this part
                early_stop = False
                with torch.no_grad():
                    self.metric.clear()
                    for data in validate_loader:
                        info = self._validate_step(data)
                    self.metric.display()
                    if self.callbacks:
                        for callback in self.callbacks:
                            try:
                                callback.step(self.metric.cur_metric)
                            except StopIteration:
                                early_stop = True
                                checkpoint_path = callback.path

                if early_stop:
                    self.model.load_state_dict(torch.load(checkpoint_path))
                    print('early stop at epoch {}'.format(epoch))
                    break

    def _train_step(self, data):
        # forward batch
        data = move_to_device(data)
        y_pred, y_true = self.forward_batch(data)
        # compute loss
        loss = self.criterion(y_pred, y_true)
        # update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # summary info in train step
        info = {'loss': loss.item()}
        return info

    def predict(self, test_loader):
        output = []
        with torch.no_grad():
            for data in test_loader:
                data = move_to_device(data)
                y_out = self.predict_batch(data)
                output.append(y_out)
        return torch.stack(output)

    def _validate_step(self, data):
        data = move_to_device(data)
        y_pred, y_true = self.forward_batch(data)
        self.metric.update_state(y_pred, y_true)
        info = {'metric': self.metric.cur_metric}
        return info

    def forward_batch(self, data):
        pass

    def predict_batch(self, data):
        pass


