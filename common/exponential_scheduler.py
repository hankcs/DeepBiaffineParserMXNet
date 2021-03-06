# -*- coding:utf-8 -*-
# Author：hankcs
# Date: 2018-09-02 16:28
from mxnet.lr_scheduler import LRScheduler


class ExponentialScheduler(LRScheduler):
    def __init__(self, base_lr=0.01, decay_rate=0.5, decay_every=1, warmup_steps=0, warmup_begin_lr=0,
                 warmup_mode='linear'):
        super().__init__(base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
        self.decay_rate = decay_rate
        self.decay_every = decay_every

    def __call__(self, num_update):
        return self.base_lr * self.decay_rate ** (num_update / self.decay_every)
