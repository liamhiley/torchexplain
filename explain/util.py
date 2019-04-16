import torch
from torch import nn
import torch.nn.functional as F
from autograd import avg_pool3d
from .conv import Conv3d
avg_pool3d = avg_pool3d.apply
from functools import partial
class Shortcut(nn.Module):
    def forward(self, *input):
        assert len(input) > 1
        return (torch.cat(tuple(input))).sum(dim=0,keepdim=True)

class Flatten(nn.Module):
    """
    Utility class for PyTorch models, please add this in place of any flattening you use in your model for use with LRP
    """
    def forward(self, input):
        return input.view(input.size(0), -1)

class Downsample(nn.Module):
    """
    Utility class for PyTorch models, please add this in place of any downsampling you use in your model for use with LRP.
    Models with shortcut connections (ResNET, DenseNET, Inception) may use this.
    """
    def __init__(self, in_, out_, mode="A", **kwargs):
        """
        :param out: The desired output size/shape for the downsample to result in.
        :param mode: A for Average Pooling followed by zero padding, B for Convolution followed by Batch Normalisation
        """
        super(Downsample, self).__init__()
        self.dim = kwargs.pop("dim", 2)
        self.kernel = kwargs.pop("kernel", 1)
        self.stride = kwargs.pop("stride", 1)

        if self.dim == 1:
            conv_layer = nn.Conv1d
            pool = nn.avg_pool1d
            batch_layer = nn.BatchNorm1d
        elif self.dim == 2:
            conv_layer = nn.Conv2d
            pool = nn.avg_pool2d
            batch_layer = nn.BatchNorm2d
        elif self.dim == 3:
            conv_layer = Conv3d
            pool = avg_pool3d
            batch_layer = nn.BatchNorm3d

        def basic(x, out):
            y = pool(x, 1, self.stride)
            zero_pads = torch.Tensor(
                y.size(0), out - y.size(1), y.size(2), y.size(3),
                y.size(4)
            ).zero_().requires_grad_()
            if torch.cuda.is_available() and isinstance(y.data, torch.cuda.FloatTensor):
                zero_pads = zero_pads.cuda()
            y = torch.cat([y, zero_pads], dim=1)
            return y

        def conv(x):
            return nn.Sequential(
                conv_layer(in_, out_, stride=self.stride),
                batch_layer(out_)
            )

        self.mode = partial(basic, out=out_) if mode=="A" else conv
    def forward(self, input):
        return self.mode(input)