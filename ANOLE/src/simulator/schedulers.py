from typing import List

import numpy as np

from simulator.master_QoE_4G.abr_trace import AbrTrace


class Scheduler:
    def __init__(self):
        self.epoch = 0

    def get_trace(self):
        raise NotImplementedError

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class TestScheduler(Scheduler):
    def __init__(self, trace: AbrTrace):
        super().__init__()
        self.trace = trace

    def get_trace(self):
        return self.trace


class UDRTrainScheduler(Scheduler):
    def __init__(
        self, traces: List[AbrTrace], percent: float = 1.0
    ):
        super().__init__()
        self.traces = traces
        self.percent = percent

    def get_trace(self):
        return np.random.choice(self.traces)




