from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class Timing:
    start: torch.cuda.Event
    end: torch.cuda.Event
    time: float = None

    def __call__(self):
        if self.time is None:
            self.end.synchronize()
            self.time = self.start.elapsed_time(self.end)
        return self.time


@contextmanager
def cuda_timing():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    yield Timing(start, end)
    end.record()
