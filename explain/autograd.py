import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
import torch.autograd as autograd
from torch.autograd import Function
import copy

class conv3d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, range=None):
        # torch.Size([1, 64, 16, 56, 56])
        ctx.save_for_backward(input, weight, bias)
        ctx.hparam = [stride, padding, dilation, groups, range]
        return F.conv3d(input, weight, bias, stride,
                        padding, dilation, groups)
    @staticmethod
    def backward(ctx, grad_output):
        import pydevd
        pydevd.settrace(suspend=False, trace_only_current_thread=True)
        # input (N,C,D,H,W)
        # weights (F,C,DD,HH,WW)
        # output (N,F,Hd,Hh,Hw)
        input, weights, bias = ctx.saved_tensors
        stride, padding, dilation, groups, range = ctx.hparam
        if range:
            pweights = nweights = weights
            pweights = torch.clamp(weights, min=0)
            nweights = torch.clamp(weights, max=0)
            lowest, highest = range
            linput = lowest + 0 * input.data
            hinput = highest + 0 * input.data
            output = F.conv3d(input, weights, None, stride,
                              padding, dilation, groups)
            pout = F.conv3d(linput, pweights, None, stride,
                              padding, dilation, groups)
            nout = F.conv3d(hinput, nweights, None, stride,
                              padding, dilation, groups)

            root_out = output - pout - nout
            norm_grad = grad_output / root_out
            norm_grad[root_out==0] = 0
            grad = torch.nn.grad.conv3d_input(input.shape, weights, norm_grad, stride=stride, padding=padding)
            pgrad = torch.nn.grad.conv3d_input(input.shape, pweights, norm_grad, stride=stride, padding=padding)
            ngrad = torch.nn.grad.conv3d_input(input.shape, nweights, norm_grad, stride=stride, padding=padding)
            if all(ctx.needs_input_grad):
                weights.grad = torch.nn.grad.conv3d_weight(input, weights.shape, norm_grad, stride=stride,
                                                           padding=padding)
                if bias is not None:
                    bias.grad = torch.nn.grad.conv3d_weight(input, bias.shape, norm_grad, stride=stride,
                                                            padding=padding)
                    return (input.grad * input - pgrad * linput - ngrad * hinput), weights.grad, bias.grad, None, None, None, None, None
                return (input.grad * input - pgrad * linput - ngrad * hinput), weights.grad, None, None, None, None, None, None
            return (input.grad * input - pgrad * linput - ngrad * hinput), None, None, None, None, None, None, None
        else:
            weights = torch.clamp(weights, min=0)
            output = F.conv3d(input, weights, None, stride,
                            padding, dilation, groups)
            norm_grad = grad_output / output
            # following EB special case, zero outputs result in relevance of 0 in grad
            norm_grad[output==0] = 0

            input.grad = torch.nn.grad.conv3d_input(input.shape, weights, norm_grad, stride=stride, padding=padding)
            if all(ctx.needs_input_grad):
                weights.grad = torch.nn.grad.conv3d_weight(input, weights.shape, norm_grad, stride=stride,
                                                    padding=padding)
                if bias is not None:
                    bias.grad = torch.nn.grad.conv3d_weight(input, bias.shape, norm_grad, stride=stride,
                                                    padding=padding)
                    return (input.grad * input), weights.grad, bias.grad, None, None, None, None, None
                return (input.grad * input), weights.grad, None, None, None, None, None, None
            return (input.grad * input), None, None, None, None, None, None, None

class avg_pool3d(Function):
    @staticmethod
    def forward(ctx, input, kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True):
        # if len(stride) == 1:
        #     # view 1d convolutions as 2d
        #     stride = (1,) + stride
        #     padding = (0,) + padding
        #     dilation = (1,) + dilation
        output = F.avg_pool3d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
        ctx.save_for_backward(input, output)
        ctx.hparams = [_triple(kernel_size), _triple(stride), _triple(padding), ceil_mode, count_include_pad]
        return output
    @staticmethod
    def backward(ctx, grad_output):
        import pydevd
        pydevd.settrace(suspend=False, trace_only_current_thread=True)
        input, output = ctx.saved_tensors
        kernel_size, stride, padding, ceil_mode, count_include_pad = ctx.hparams
        norm_grad = grad_output / output
        # following EB special case, zero outputs result in relevance of 0 in grad
        norm_grad[output==0] = 0
        pad = []
        for p in _triple(padding):
            pad1 = p // 2
            pad2 = p - pad1
            pad += [pad1, pad2]
        padding = tuple(pad)
        pad_input = F.pad(input, pad)
        N, C, D, H, W = output.shape
        padded_grad = torch.zeros_like(pad_input)
        # The gradient of each element of the output can be distributed equally to each of the elements in the padded input pooling block
        for n in range(N):
            for c in range(C):
                for d in range(0,D,stride[0]):
                    for h in range(0,H,stride[1]):
                        for w in range(0,W,stride[2]):
                            padded_grad[n,c,d:d+kernel_size[0],h:h+kernel_size[1],w:w+kernel_size[2]] += norm_grad[n,c,d,h,w]/torch.Tensor(kernel_size).prod(0)
        n_pad = [0 for _ in padding]
        for i in range(0, 6, 2):
            if padding[i] > 0:
                n_pad[i] = padding[i]
            if padding[i + 1] > 0:
                n_pad[i + 1] = padding[i]
            else:
                n_pad[i + 1] = input.shape[-1 - i // 2] + 1
        padding = tuple(n_pad)
        input.grad = padded_grad[:,:,padding[-2]:padding[-1], padding[-4]:padding[-3], padding[-6]:padding[-5]]
        return (input * input.grad), None, None, None, None, None

class max_pool3d(Function):
    @staticmethod
    def forward(ctx, input, kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False, return_indices=True):
        output, return_indices = F.max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
        ctx.save_for_backward(input, output)
        ctx.hparams = [kernel_size, stride, padding, dilation, ceil_mode, return_indices]
        return output
    @staticmethod
    def backward(ctx, grad_output):
        import pydevd
        pydevd.settrace(suspend=False, trace_only_current_thread=True)
        input, output = ctx.saved_tensors
        kernel_size, stride, padding, dilation, ceil_mode, return_indices = ctx.hparams
        norm_grad = grad_output / output
        # following EB special case, zero outputs result in relevance of 0 in grad
        norm_grad[output==0] = 0
        # The gradient of each element of the output is attributed in a winner takes all strategy, to the argmax of the input
        input.grad = F.max_unpool3d(norm_grad,return_indices,kernel_size,stride,padding,input.shape)
        return (input * input.grad), None, None, None, None, None, None

class linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
        ctx.save_for_backward(input, weight, bias)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        import pydevd
        pydevd.settrace(suspend=False, trace_only_current_thread=True)
        input, weight, bias = ctx.saved_tensors
        weight = torch.clamp(weight, min=0)
        output = input.matmul(weight.t())
        norm_grad = grad_output / output
        # following EB special case, zero outputs result in relevance of 0 in grad
        norm_grad[output==0] = 0
        input.grad = norm_grad.matmul(weight)
        return (input * input.grad), None, None

class threshold(Function):
    @staticmethod
    def forward(ctx, input, threshold, value, inplace=False):
        return F.threshold(input,threshold,value, inplace)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None