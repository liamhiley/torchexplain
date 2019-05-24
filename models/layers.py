# Adapted from https://github.com/greydanus/excitationbp/blob/master/excitationbp/

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _single, _pair, _triple
from functools import partial

_thnn_convs = {}


class conv3d(Function):
    def __init__(self, stride, padding, dilation, transposed, output_padding, groups):
        super(conv3d, self).__init__()
        if len(stride) == 1:
            # view 1d convolutions as 2d
            stride = (1,) + stride
            padding = (0,) + padding
            dilation = (1,) + dilation
            output_padding = (0,) + output_padding
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, input, weight, bias=None):
        # torch.Size([1, 64, 16, 56, 56])
        self.save_for_backward(input, weight, bias)
        return F.conv3d(input, weight.shape, self.dilation, self.padding, self.stride)

    def backward(self, grad_output):
        k = grad_output.dim()
        grad_output = grad_output.contiguous()
        input, weight, bias = self.saved_tensors
        input = input.contiguous()

        ### start EB-SPECIFIC CODE  ###
        # print("this is a {} conv layer ({})"
        #       .format('pos' if torch.use_pos_weights else 'neg', grad_output.sum()))
        weight = weight.clamp(min=0) if torch.use_pos_weights else weight.clamp(max=0).abs()
        bias = bias.clamp(min=0) if torch.use_pos_weights else bias.clamp(max=0).abs()

        # a mini forward pass, using the positive weights
        input = input - input.min() if input.min() < 0 else input
        k = input.dim()
        input = input.contiguous()
        if k == 3:
            input, weight = _view4d(input, weight)
        norm_factor = self._update_output(input, weight, bias)
        if k == 3:
            norm_factor, = _view3d(norm_factor)

        grad_output /= norm_factor + 1e-20  # normalize
        ### stop EB-SPECIFIC CODE  ###

        if k == 3:
            grad_output, input, weight_ = _view4d(grad_output, input, weight)

        grad_input = (self._grad_input(input, weight, grad_output))

        grad_weight, grad_bias = (
            self._grad_params(input, weight, bias, grad_output)
            if any(self.needs_input_grad[1:]) else (None, None))
        if k == 3:
            grad_input, grad_weight, = _view3d(grad_input, grad_weight)

        ### start EB-SPECIFIC CODE  ###
        grad_input *= input
        #         grad_input *= 1/grad_input.sum() # extra normalization...not rigorous
        ### stop EB-SPECIFIC CODE  ###

        return grad_input, grad_weight, grad_bias



class LRPLinear(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables

        ### start EB-SPECIFIC CODE  ###
        # print("this is a {} linear layer ({})"
        #       .format('pos' if torch.use_pos_weights else 'neg', grad_output.sum().data[0]))
        weight = weight.clamp(min=0) if torch.use_pos_weights else weight.clamp(max=0).abs()

        input.data = input.data - input.data.min() if input.data.min() < 0 else input.data
        grad_output /= input.mm(weight.t()).abs() + 1e-10 # normalize
        ### stop EB-SPECIFIC CODE  ###

        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
            ### start EB-SPECIFIC CODE  ###
            grad_input *= input
            ### stop EB-SPECIFIC CODE  ###

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)


        return grad_input, grad_weight, grad_bias


def lrp_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    f = LRPConv(_triple(stride), _triple(padding), _triple(dilation), False, _triple(0), groups)
    if bias is None:
        return f(input, weight)
    return f(input, weight, bias)

def lrp(switch):
    global root_funcs
    if switch:
        torch.nn.functional.conv3d = lrp_conv3d
        torch.nn.functional.linear = LRPLinear.apply


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
            pool = F.avg_pool1d
            batch_layer = nn.BatchNorm1d
        elif self.dim == 2:
            conv_layer = nn.Conv2d
            pool = F.avg_pool2d
            batch_layer = nn.BatchNorm2d
        elif self.dim == 3:
            conv_layer = nn.Conv3d
            pool = F.avg_pool3d
            batch_layer = nn.BatchNorm3d

        def basic(x, out):
            y = F.avg_pool3d(x, kernel_size=1, stride=self.stride)
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

def _view4d(*tensors):
    # view 3d tensor as 4d (conv1d as conv2d)
    output = []
    for t in tensors:
        assert t.dim() == 3
        size = list(t.size())
        size.insert(2, 1)
        output += [t.view(*size)]
    return output


def _view3d(*tensors):
    # view 4d tensor as 3d
    output = []
    for t in tensors:
        if t is None:
            output += [None]
        else:
            assert t.dim() == 4 and t.size(2) == 1
            output += [t.squeeze(2)]
    return output