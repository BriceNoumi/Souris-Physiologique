import scipy.signal
import numpy as np
from collections import deque

DataSample = float | int

class Filter:
    """
    Abstract class for digital filters

    Digital filters are objects that can be called like functions to
    filter a single sample of data. The filter object should store
    any necessary state information between calls.

    (Note: this implies that the filtering operation is not idempotent)
    """
    def __call__(self, x: DataSample) -> DataSample:
        raise NotImplementedError
    
class ButterworthLPF(Filter):
    """
    Butterworth low-pass filter

    (Credit: https://www.samproell.io/posts/yarppg/yarppg-live-digital-filter/)
    """
    def __init__(self, cutoff: float, fs: float, order: int):
        self.b, self.a = scipy.signal.butter(order, cutoff, fs=fs, btype='low')

        assert(self.a[0] == 1)

        alen = len(self.a) - 1
        self._xh = deque([0] * len(self.b), maxlen=len(self.b))
        self._yh = deque([0] * alen, maxlen=alen)
        
    def __call__(self, x: DataSample) -> DataSample:
        self._xh.appendleft(x)
        y = np.dot(self.b, self._xh) - np.dot(self.a[1:], self._yh)
        self._yh.appendleft(y)
        return y


