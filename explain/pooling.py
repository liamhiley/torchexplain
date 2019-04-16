import torch
import torch.nn as nn
from .autograd import avg_pool3d, max_pool3d
from torch.nn.modules.utils import _single, _pair, _triple

avg_pool3d = avg_pool3d.apply
max_pool3d = max_pool3d.apply

class _AvgPoolNd(nn.Module):
    def extra_repr(self):
        return 'kernel_size={}, stride={}, padding={}'.format(
            self.kernel_size, self.stride, self.padding
        )

class AvgPool3d(_AvgPoolNd):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        super(AvgPool3d, self).__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride) or _triple(kernel_size)
        self.padding = _triple(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
    def forward(self, input):
        return avg_pool3d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)

class _MaxPoolNd(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=True):
        super(_MaxPoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
            ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)

class MaxPool3d(_MaxPoolNd):
    def forward(self, input):
        return max_pool3d(input, _triple(self.kernel_size), _triple(self.stride),
                            _triple(self.padding), _triple(self.dilation), self.ceil_mode,
                            self.return_indices)