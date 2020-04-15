import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
import torch.autograd as autograd
from torch.autograd import Function
import torchexplain
torchexplain.shortcut = None
import copy

class conv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, shortcut=False):
        # torch.Size([1, 64, 16, 56, 56])
        ctx.save_for_backward(input, weight, bias)
        ctx.hparam = [stride, padding, dilation, groups, shortcut]
        return F.conv2d(input, weight, bias, stride,
                        padding, dilation, groups)
    @staticmethod
    def backward(ctx, grad_output):
        # input (N,C,D,H,W)
        # weights (F,C,DD,HH,WW)
        # output (N,F,Hd,Hh,Hw)
        input, weights, bias = ctx.saved_tensors
        stride, padding, dilation, groups, shortcut = ctx.hparam
        if (torchexplain.shortcut == "shortcut" and shortcut) or (torchexplain.shortcut == "main" and not shortcut) or torchexplain.shortcut is None:
            weights = torch.clamp(weights, min=0)
        else:
            # essentially turn off this connection by assigning equal contribution to all neurons
            weights = torch.ones_like(weights)
        output = F.conv2d(input, weights, None, stride,
                        padding, dilation, groups)
        output[output==0] += torchexplain.epsilon
        norm_grad = grad_output / output
        # following EB special case, zero outputs result in relevance of 0 in grad
        if not torchexplain.epsilon:
            norm_grad[output==0] = 0

        input.grad = torch.nn.grad.conv2d_input(input.shape, weights, norm_grad, stride=stride, padding=padding)
        if all(ctx.needs_input_grad):
            weights.grad = torch.nn.grad.conv2d_weight(input, weights.shape, norm_grad, stride=stride,
                                                padding=padding)
            if bias is not None:
                bias.grad = torch.nn.grad.conv2d_weight(input, bias.shape, norm_grad, stride=stride,
                                                padding=padding)
                return (input.grad * input), weights.grad, bias.grad, None, None, None, None, None
            return (input.grad * input), weights.grad, None, None, None, None, None, None
        return (input.grad * input), None, None, None, None, None, None, None

class conv3d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, shortcut=False):
        # torch.Size([1, 64, 16, 56, 56])

        ctx.hparam = [stride, padding, dilation, groups, shortcut]
        output = F.conv3d(input, weight, bias, stride,
                        padding, dilation, groups)
        ctx.save_for_backward(input, weight, bias)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        # input (N,C,D,H,W)
        # weights (F,C,DD,HH,WW)
        # output (N,F,Hd,Hh,Hw)
        input, weights, bias = ctx.saved_tensors
        stride, padding, dilation, groups, shortcut = ctx.hparam
        if (torchexplain.shortcut == "shortcut" and shortcut) or (
                torchexplain.shortcut == "main" and not shortcut) or torchexplain.shortcut is None:
            weights = torch.clamp(weights, min=0)
        else:
            # essentially turn off this connection by assigning equal contribution to all neurons
            weights = torch.ones_like(weights)
        output = F.conv3d(input, weights, None, stride,
                          padding, dilation, groups)
        output[output == 0] += torchexplain.epsilon
        norm_grad = grad_output / output
        # following EB special case, zero outputs result in relevance of 0 in grad
        if not torchexplain.epsilon:
            norm_grad[output == 0] = 0

        input.grad = torch.nn.grad.conv3d_input(input.shape, weights, norm_grad, stride=stride, padding=padding, groups=groups)
        if all(ctx.needs_input_grad):
            weights.grad = torch.nn.grad.conv3d_weight(input, weights.shape, norm_grad, stride=stride,
                                                       padding=padding)
            if bias is not None:
                bias.grad = torch.nn.grad.conv3d_weight(input, bias.shape, norm_grad, stride=stride,
                                                        padding=padding)
                return (input.grad * input), weights.grad, bias.grad, None, None, None, None, None
            return (input.grad * input), weights.grad, None, None, None, None, None, None
        return (input.grad * input), None, None, None, None, None, None, None

class firstconv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, range=None):
        ctx.save_for_backward(input, weight, bias)
        ctx.hparam = [stride, padding, dilation, groups, range]
        return F.conv2d(input, weight, bias, stride,
                        padding, dilation, groups)
    @staticmethod
    def backward(ctx, grad_output):
        input, weights, bias = ctx.saved_tensors
        stride, padding, dilation, groups, range = ctx.hparam
        pweights = nweights = weights
        pweights = torch.clamp(weights, min=0)
        nweights = torch.clamp(weights, max=0)
        lowest, highest = range
        linput = lowest + 0 * input.data
        hinput = highest + 0 * input.data
        output = F.conv2d(input, weights, None, stride,
                          padding, dilation, groups)
        pout = F.conv2d(linput, pweights, None, stride,
                          padding, dilation, groups)
        nout = F.conv2d(hinput, nweights, None, stride,
                          padding, dilation, groups)

        root_out = output - pout - nout
        root_out[root_out==0] += torchexplain.epsilon
        norm_grad = grad_output / root_out
        if not torchexplain.epsilon:
            norm_grad[root_out==0] = 0
        grad = torch.nn.grad.conv2d_input(input.shape, weights, norm_grad, stride=stride, padding=padding)
        pgrad = torch.nn.grad.conv2d_input(input.shape, pweights, norm_grad, stride=stride, padding=padding)
        ngrad = torch.nn.grad.conv2d_input(input.shape, nweights, norm_grad, stride=stride, padding=padding)
        if all(ctx.needs_input_grad):
            weights.grad = torch.nn.grad.conv2d_weight(input, weights.shape, norm_grad, stride=stride,
                                                       padding=padding)
            if bias is not None:
                bias.grad = torch.nn.grad.conv2d_weight(input, bias.shape, norm_grad, stride=stride,
                                                        padding=padding)
                return (grad * input - pgrad * linput - ngrad * hinput), weights.grad, bias.grad, None, None, None, None, None
            return (grad * input - pgrad * linput - ngrad * hinput), weights.grad, None, None, None, None, None, None
        return (grad * input - pgrad * linput - ngrad * hinput), None, None, None, None, None, None, None

class firstconv3d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, range=None):

        ctx.hparam = [stride, padding, dilation, groups, range]
        output = F.conv3d(input, weight, bias, stride,
                        padding, dilation, groups)
        ctx.save_for_backward(input, weight, bias, output)
        return output
    @staticmethod
    def backward(ctx, grad_output):

        input, weights, bias, output = ctx.saved_tensors
        stride, padding, dilation, groups, range = ctx.hparam
        pweights = torch.clamp(weights, min=0)
        nweights = torch.clamp(weights, max=0)
        lowest, highest = range
        linput = lowest + 0 * input.data
        hinput = highest + 0 * input.data
        # output = F.conv3d(input, weights, None, stride,
                          # padding, dilation, groups)
        pout = F.conv3d(linput, pweights, None, stride,
                        padding, dilation, groups)
        nout = F.conv3d(hinput, nweights, None, stride,
                        padding, dilation, groups)

        root_out = output - pout - nout
        root_out[root_out == 0] += torchexplain.epsilon
        norm_grad = grad_output / root_out
        if not torchexplain.epsilon:
            norm_grad[root_out == 0] = 0
        grad = torch.nn.grad.conv3d_input(input.shape, weights, norm_grad, stride=stride, padding=padding, groups=groups)
        pgrad = torch.nn.grad.conv3d_input(input.shape, pweights, norm_grad, stride=stride, padding=padding, groups=groups)
        ngrad = torch.nn.grad.conv3d_input(input.shape, nweights, norm_grad, stride=stride, padding=padding, groups=groups)
        if all(ctx.needs_input_grad):
            weights.grad = torch.nn.grad.conv3d_weight(input, weights.shape, norm_grad, stride=stride,
                                                       padding=padding)
            if bias is not None:
                bias.grad = torch.nn.grad.conv3d_weight(input, bias.shape, norm_grad, stride=stride,
                                                        padding=padding)
                return (grad * input - pgrad * linput - ngrad * hinput), weights.grad, bias.grad, None, None, None, None, None
            return (grad * input - pgrad * linput - ngrad * hinput), weights.grad, None, None, None, None, None, None
        return (grad * input - pgrad * linput - ngrad * hinput), None, None, None, None, None, None, None

class abconv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, beta=0, shortcut=False):
        ctx.save_for_backward(input, weight, bias)
        ctx.hparam = [stride, padding, dilation, groups, alpha, beta, shortcut]
        return F.conv2d(input, weight, bias, stride,
                        padding, dilation, groups)
    @staticmethod
    def backward(ctx, grad_output):
        input, weights, bias = ctx.saved_tensors
        stride, padding, dilation, groups, alpha, beta, shortcut = ctx.hparam

        pinput = torch.clamp(input, min=0)
        linput = torch.clamp(input, max=0)

        if (torchexplain.shortcut == "shortcut" and shortcut) or (
                torchexplain.shortcut == "main" and not shortcut) or torchexplain.shortcut is None:
            weights = torch.clamp(weights, min=0)
            pweights = nweights = weights
            pweights = torch.clamp(weights, min=0)
            nweights = torch.clamp(weights, max=0)
        else:
            # essentially turn off this connection by assigning equal contribution to all neurons
            weights = torch.ones_like(weights)
            pweights = nweights = weights


        pout = F.conv2d(pinput, pweights, None, stride,
                        padding, dilation, groups)
        nout = F.conv2d(linput, nweights, None, stride,
                        padding, dilation, groups)
        sum_out = pout + nout
        sum_out[sum_out==0] += 1e-9

        norm_grad = grad_output / sum_out

        # norm_grad[sum_out == 0] = 0

        agrad = torch.nn.grad.conv2d_input(input.shape, pweights, norm_grad, stride=stride, padding=padding, groups=groups)
        agrad *= pinput
        bgrad = torch.nn.grad.conv2d_input(input.shape, nweights, norm_grad, stride=stride, padding=padding, groups=groups)
        bgrad *= linput

        grad = agrad + bgrad

        if beta:
            cpout = F.conv2d(pinput, nweights, None, stride,
                            padding, dilation, groups)
            cnout = F.conv2d(linput, pweights, None, stride,
                            padding, dilation, groups)
            csum_out = cpout + cnout
            csum_out[csum_out==0] += torchexplain.epsilonlon
            c_grad = grad_output / csum_out
            if not torchexplain.epsilonlon:
                c_grad[csum_out == 0] = 0

            c_agrad = torch.nn.grad.conv2d_input(input.shape, nweights, c_grad, stride=stride, padding=padding, groups=groups)
            c_agrad *= pinput
            c_bgrad = torch.nn.grad.conv2d_input(input.shape, pweights, c_grad, stride=stride, padding=padding, groups=groups)
            c_bgrad *= linput

            c_grad = c_agrad + c_bgrad

            grad = alpha * grad - beta * c_grad
        if all(ctx.needs_input_grad):
            weights.grad = torch.nn.grad.conv2d_weight(input, weights.shape, grad, stride=stride,padding=padding)
            if bias is not None:
                bias.grad = torch.nn.grad.conv2d_weight(input, bias.shape, grad, stride=stride,padding=padding)
                return grad, weights.grad, bias.grad, None, None, None, None, None, None, None
            return grad, weights.grad, None, None, None, None, None, None, None, None
        return grad, None, None, None, None, None, None, None, None, None

class abconv3d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, beta=0, shortcut=False):
        ctx.save_for_backward(input, weight, bias)
        ctx.hparam = [stride, padding, dilation, groups, alpha, beta, shortcut]
        return F.conv3d(input, weight, bias, stride,
                        padding, dilation, groups)
    @staticmethod
    def backward(ctx, grad_output):
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        input, weights, bias = ctx.saved_tensors
        stride, padding, dilation, groups, alpha, beta, shortcut = ctx.hparam

        pinput = torch.clamp(input, min=0)
        linput = torch.clamp(input, max=0)

        if (torchexplain.shortcut == "shortcut" and shortcut) or (
                torchexplain.shortcut == "main" and not shortcut) or torchexplain.shortcut is None:
            weights = torch.clamp(weights, min=0)
            pweights = nweights = weights
            pweights = torch.clamp(weights, min=0)
            nweights = torch.clamp(weights, max=0)
        else:
            # essentially turn off this connection by assigning equal contribution to all neurons
            weights = torch.ones_like(weights)
            pweights = nweights = weights

        pout = F.conv3d(pinput, pweights, None, stride,
                        padding, dilation, groups)
        nout = F.conv3d(linput, nweights, None, stride,
                        padding, dilation, groups)
        sum_out = pout + nout
        sum_out[sum_out == 0] += 1e-9

        norm_grad = grad_output / sum_out

        # norm_grad[sum_out == 0] = 0

        agrad = torch.nn.grad.conv3d_input(input.shape, pweights, norm_grad, stride=stride, padding=padding, groups=groups)

        agrad *= pinput
        bgrad = torch.nn.grad.conv3d_input(input.shape, nweights, norm_grad, stride=stride, padding=padding, groups=groups)
        bgrad *= linput

        grad = agrad + bgrad
        if beta:
            cpout = F.conv3d(pinput, nweights, None, stride,
                             padding, dilation, groups)
            cnout = F.conv3d(linput, pweights, None, stride,
                             padding, dilation, groups)
            csum_out = cpout + cnout
            csum_out[csum_out == 0] += torchexplain.epsilonlon
            c_grad = grad_output / csum_out
            if not torchexplain.epsilonlon:
                c_grad[csum_out == 0] = 0

            c_agrad = torch.nn.grad.conv3d_input(input.shape, nweights, c_grad, stride=stride, padding=padding, groups=groups)
            c_agrad *= pinput
            c_bgrad = torch.nn.grad.conv3d_input(input.shape, pweights, c_grad, stride=stride, padding=padding, groups=groups)
            c_bgrad *= linput

            c_grad = c_agrad + c_bgrad

            grad = alpha * grad - beta * c_grad
        if all(ctx.needs_input_grad):
            weights.grad = torch.nn.grad.conv3d_weight(input, weights.shape, grad, stride=stride, padding=padding, groups=groups)
            if bias is not None:
                bias.grad = torch.nn.grad.conv3d_weight(input, bias.shape, grad, stride=stride, padding=padding, groups=groups)
                return grad, weights.grad, bias.grad, None, None, None, None, None, None, None
            return grad, weights.grad, None, None, None, None, None, None, None, None
        return grad, None, None, None, None, None, None, None, None, None

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
                            # padded_grad[n, c, d:d + kernel_size[0], h:h + kernel_size[1], w:w + kernel_size[2]] += 1 / torch.tensor(kernel_size).prod(0)
                            padded_grad[n,c,d:d+kernel_size[0],h:h+kernel_size[1],w:w+kernel_size[2]] += norm_grad[n,c,d,h,w]/torch.tensor(kernel_size).prod(0)
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

class avg_pool2d(Function):
    @staticmethod
    def forward(ctx, input, kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True):
        # if len(stride) == 1:
        #     # view 1d convolutions as 2d
        #     stride = (1,) + stride
        #     padding = (0,) + padding
        #     dilation = (1,) + dilation
        output = F.avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
        ctx.save_for_backward(input, output)
        ctx.hparams = [_pair(kernel_size), _pair(stride), _pair(padding), ceil_mode, count_include_pad]
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        kernel_size, stride, padding, ceil_mode, count_include_pad = ctx.hparams
        norm_grad = grad_output / output
        # following EB special case, zero outputs result in relevance of 0 in grad
        norm_grad[output==0] = 0
        pad = []
        for p in _pair(padding):
            pad1 = p // 2
            pad2 = p - pad1
            pad += [pad1, pad2]
        padding = tuple(pad)
        pad_input = F.pad(input, pad)
        N, C, H, W = output.shape
        padded_grad = torch.zeros_like(pad_input)
        # The gradient of each element of the output can be distributed equally to each of the elements in the padded input pooling block
        for n in range(N):
            for c in range(C):
                for h in range(0,H,stride[0]):
                    for w in range(0,W,stride[1]):
                        # padded_grad[n, c, d:d + kernel_size[0], h:h + kernel_size[1], w:w + kernel_size[2]] += 1 / torch.tensor(kernel_size).prod(0)
                        padded_grad[n,c,h:h+kernel_size[0],w:w+kernel_size[1]] += norm_grad[n,c,h,w]/torch.tensor(kernel_size).prod(0)
        n_pad = [0 for _ in padding]
        for i in range(0, 4, 2):
            if padding[i] > 0:
                n_pad[i] = padding[i]
            if padding[i + 1] > 0:
                n_pad[i + 1] = padding[i]
            else:
                n_pad[i + 1] = input.shape[-1 - i // 2] + 1
        padding = tuple(n_pad)
        input.grad = padded_grad[:,:,padding[-2]:padding[-1], padding[-4]:padding[-3]]
        return (input * input.grad), None, None, None, None, None

class adaptive_avg_pool3d(Function):
    @staticmethod
    def forward(ctx, input, output_size):
        output = F.adaptive_avg_pool3d(input, output_size)
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        # Reverse engineer params
        x_size = torch.tensor(input.size(), dtype=torch.int)[1:]
        y_size = torch.tensor(output.size(), dtype=torch.int)[1:]
        if torch.any(y_size == 0):
            zero_idcs = y_size == 0
            y_size[zero_idcs] += 1
            kernel_size = (x_size + y_size - 1) // y_size
            kernel_size[zero_idcs] = 0
        else:
            kernel_size = (x_size + y_size - 1) // y_size
        if torch.any(y_size == 1):
            ones_idcs = y_size == 1
            y_size[ones_idcs] += 1
            stride = (x_size - kernel_size) // (y_size - 1)
            stride[ones_idcs] = 0
        else:
            stride = (x_size - kernel_size)
        kernel_size, stride = tuple([ele.item() for ele in kernel_size]), tuple([ele.item() for ele in stride])
        padding = (0, 0, 0)
        norm_grad = grad_output / output
        # following EB special case, zero outputs result in relevance of 0 in grad
        norm_grad[output == 0] = 0
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
                for d in range(0, D, stride[0]):
                    for h in range(0, H, stride[1]):
                        for w in range(0, W, stride[2]):
                            # padded_grad[n, c, d:d + kernel_size[0], h:h + kernel_size[1], w:w + kernel_size[2]] += 1 / torch.tensor(kernel_size).prod(0)
                            padded_grad[n, c, d:d + kernel_size[0], h:h + kernel_size[1], w:w + kernel_size[2]] += norm_grad[n, c, d, h, w] / torch.tensor(kernel_size).prod(0)
        n_pad = [0 for _ in padding]
        for i in range(0, 6, 2):
            if padding[i] > 0:
                n_pad[i] = padding[i]
            if padding[i + 1] > 0:
                n_pad[i + 1] = padding[i]
            else:
                n_pad[i + 1] = input.shape[-1 - i // 2] + 1
        padding = tuple(n_pad)
        input.grad = padded_grad[:, :, padding[-2]:padding[-1], padding[-4]:padding[-3], padding[-6]:padding[-5]]
        return (input * input.grad), None, None, None, None, None

class adaptive_avg_pool2d(Function):
    @staticmethod
    def forward(ctx, input, output_size):
        output = F.adaptive_avg_pool2d(input, output_size)
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        input, output = ctx.saved_tensors
        # Reverse engineer params
        x_size = torch.tensor(input.size(), dtype=torch.int)[1:]
        y_size = torch.tensor(output.size(), dtype=torch.int)[1:]
        if torch.any(y_size == 0):
            zero_idcs = y_size == 0
            y_size[zero_idcs] += 1
            kernel_size = (x_size + y_size - 1) // y_size
            kernel_size[zero_idcs] = 0
        else:
            kernel_size = (x_size + y_size - 1) // y_size
        if torch.any(y_size == 1):
            ones_idcs = y_size == 1
            y_size[ones_idcs] += 1
            stride = (x_size - kernel_size) // (y_size - 1)
            stride[ones_idcs] = 0
        else:
            stride = (x_size - kernel_size) // (y_size - 1)
        kernel_size, stride = tuple([ele.item() for ele in kernel_size]), tuple([ele.item() for ele in stride])
        padding = (0, 0)
        output += torchexplain.epsilon
        norm_grad = grad_output / output
        # following EB special case, zero outputs result in relevance of 0 in grad
        if not torchexplain.epsilon:
            norm_grad[output == 0] = 0
        pad = []
        for p in _pair(padding):
            pad1 = p // 2
            pad2 = p - pad1
            pad += [pad1, pad2]
        padding = tuple(pad)
        pad_input = F.pad(input, pad)
        N, C, H, W = output.shape
        padded_grad = torch.zeros_like(pad_input)
        # The gradient of each element of the output can be distributed equally to each of the elements in the padded input pooling block
        for n in range(N):
            for c in range(C):
                for h in range(0, H, stride[1]):
                    for w in range(0, W, stride[2]):
                        # padded_grad[n, c, h:h + kernel_size[0], w:w + kernel_size[1]] += 1 / torch.tensor(kernel_size).prod(0)
                        padded_grad[n, c, h:h + kernel_size[0], w:w + kernel_size[1]] += norm_grad[n, c, h, w] / torch.tensor(kernel_size).prod(0)
        n_pad = [0 for _ in padding]
        for i in range(0, 4, 2):
            if padding[i] > 0:
                n_pad[i] = padding[i]
            if padding[i + 1] > 0:
                n_pad[i + 1] = padding[i]
            else:
                n_pad[i + 1] = input.shape[-1 - i // 2] + 1
        padding = tuple(n_pad)
        input.grad = padded_grad[:, :, padding[-2]:padding[-1], padding[-4]:padding[-3]]
        return (input * input.grad), None, None, None, None, None

class max_pool2d(Function):
    @staticmethod
    def forward(ctx, input, kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False, return_indices=True):
        output, return_indices = F.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
        ctx.save_for_backward(input, output)
        ctx.hparams = [kernel_size, stride, padding, dilation, ceil_mode, return_indices]
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        kernel_size, stride, padding, dilation, ceil_mode, return_indices = ctx.hparams
        output[output==0] += torchexplain.epsilon
        norm_grad = grad_output / output
        if not torchexplain.epsilon:
            norm_grad[output == 0] = 0
        # following EB special case, zero outputs result in relevance of 0 in grad
        # norm_grad[output==0] = 0
        # The gradient of each element of the output is attributed in a winner takes all strategy, to the argmax of the input
        # input.grad = F.max_unpool2d(torch.ones_like(norm_grad), return_indices, kernel_size, stride, padding, input.shape)
        input.grad = F.max_unpool2d(norm_grad,return_indices,kernel_size,stride,padding,input.shape)
        return (input * input.grad), None, None, None, None, None, None

class max_pool3d(Function):
    @staticmethod
    def forward(ctx, input, kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False, return_indices=True):
        output, return_indices = F.max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
        ctx.save_for_backward(input, output)
        ctx.hparams = [kernel_size, stride, padding, dilation, ceil_mode, return_indices]
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        kernel_size, stride, padding, dilation, ceil_mode, return_indices = ctx.hparams
        # output += torchexplain.epsilon
        norm_grad = grad_output / output
        # following EB special case, zero outputs result in relevance of 0 in grad
        norm_grad[output==0] = 0
        # The gradient of each element of the output is attributed in a winner takes all strategy, to the argmax of the input
        # input.grad = F.max_unpool3d(torch.ones_like(norm_grad), return_indices, kernel_size, stride, padding, input.shape)
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
        input, weight, bias = ctx.saved_tensors
        weight = torch.clamp(weight, min=0)
        output = input.matmul(weight.t())
        output[output==0] += torchexplain.epsilon
        norm_grad = grad_output / output
        if not torchexplain.epsilon:
            norm_grad[output==0] = 0
        # following EB special case, zero outputs result in relevance of 0 in grad
        # norm_grad[output==0] = 0
        input.grad = norm_grad.matmul(weight)
        return (input * input.grad), None, None

class threshold(Function):
    @staticmethod
    def forward(ctx, input, threshold, value, inplace=False):
        # TODO: Generalise!
        mask = torch.zeros_like(input)
        mask[input>0] += 1.
        ctx.save_for_backward(mask)
        return F.threshold(input,threshold,value, inplace)
    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_tensors[0]
        return grad_output*mask, None, None, None

class scale(Function):
    @staticmethod
    def forward(ctx, input, min=0, range=1):
        output = torch.mul(torch.sub(input,min),1/range)
        ctx.save_for_backward(input, output)
        ctx.hparams = (min, range)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        inp, out = ctx.saved_tensors
        min, range = ctx.hparams
        return inp * grad_output/range*(inp-min)*out, None, None

class batch_norm(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
        # Training not currently supported for torchtorchexplain
        # if training:
        #     size = input.size()
        #     #
        #     #
        #     # from operator import mul
        #     # from functools import reduce
        #     #
        #     #   if reduce(mul, size[2:], size[0]) == 1
        #     size_prods = size[0]
        #     for i in range(len(size) - 2):
        #         size_prods *= size[i + 2]
        #     if size_prods == 1:
        #         raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
        output = torch.batch_norm(
            input, weight, bias, running_mean, running_var,
            training, momentum, eps, torch.backends.cudnn.enabled
        )
        ctx.save_for_backward(input, output.clone())
        ctx.hparams = (weight, bias, running_mean, running_var, eps)
        return output
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        weight, bias, running_mean, running_var, eps = ctx.hparams
        x1 = x2 = torch.zeros_like(input)
        for b in range(bias.shape[0]):
            x2[:, b, ...] = output[:, b, ...] - bias[b]
        for b in range(bias.shape[0]):
            x1[:,b,...] = input[:,b,...] - running_mean[b]
        return (input * x2)*grad_output/(x1 * output + eps), None, None, None, None, None, None, None
        # return grad_output, None, None, None, None, None, None, None

class add(Function):
    @staticmethod
    def forward(ctx, *inputs):
        sum = torch.zeros_like(inputs[0])
        for input in inputs:
            sum = sum + input
        ctx.save_for_backward(*inputs)
        return sum
    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        sum = torch.sum(torch.stack(inputs),dim=0)
        norm_grad = grad_output / sum
        for input in inputs:
            input.grad = norm_grad #/ len(inputs)

        return tuple([input * input.grad for input in inputs])
