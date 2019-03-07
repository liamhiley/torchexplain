import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import PIL.Image as pimg
import matplotlib.pyplot as plt
from skimage import transform, filters
from functools import partial
import copy
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
        dim = kwargs.pop("dim", 2)
        kernel = kwargs.pop("kernel", 1)
        stride = kwargs.pop("stride", 1)

        if dim == 1:
            conv_layer = nn.Conv1d
            pool = F.avg_pool1d
            batch_layer = nn.BatchNorm1d
        elif dim == 2:
            conv_layer = nn.Conv2d
            pool = F.avg_pool2d
            batch_layer = nn.BatchNorm2d
        elif dim == 3:
            conv_layer = nn.Conv3d
            pool = F.avg_pool3d
            batch_layer = nn.BatchNorm3d

        def basic(x, out):
            y = F.avg_pool3d(x, kernel_size=1, stride=stride)
            zero_pads = torch.Tensor(
                y.size(0), out - y.size(1), y.size(2), y.size(3),
                y.size(4)
            ).zero_()
            if torch.cuda.is_available() and isinstance(y.data, torch.cuda.FloatTensor):
                zero_pads = zero_pads.cuda()
            y = torch.cat([y.data, zero_pads], dim=1)

            return y

        def conv(x):
            return nn.Sequential(
                conv_layer(in_, out_, stride=stride),
                batch_layer(out_)
            )

        self.mode = partial(basic, out=out_, stride=stride) if mode=="A" else conv
    def forward(self, input):
        return self.mode(input)



class LRP:
    """
    Class for Layer-wise Relevance Propagation family of techniques.
    Works by finding the relevance of each layer in some way, and mapping it to the input for that layer, for each layer
    in reverse order in the model, ending with the relevance of the final layer propagated on to the input sample (image,
    signal, text)
    """
    def __init__(self, cuda):
        self.cuda = cuda

    def get_layers(self, mdl, inp=None):
        """
        :param mdl: PyTorch module with all layers registered as nn.Module objs.
        :param inp: Optional input sample to forward pass through each layer and record.
        :return layers: List of all layers used in mdl
        """
        layers = []
        if inp:
            global inputs
            inputs = []
            def save_inp(self, in_, out_):
                inputs.append(in_)
                return None
        children = list(mdl.named_children())
        if children:
            for child in children:
                layers += self.get_layers(child)
            mdl(inp)
            return layers, inputs
        else:
            if inp:
                mdl.register_forward_hook(save_inp)
            return [mdl]

    def map(self, r, **kwargs):
        if self.cuda:
            r = r.cpu()
        vis_r = r.clamp(min=0).sum((0, 1)).data.numpy()
        mode=kwargs.pop("mode",[])
        fname = kwargs.pop("fname",[])
        vid = kwargs.pop("vid",[])[[2,1,0],...]
        blur = kwargs.pop("blur",[])
        overlap = kwargs.pop("overlap",[])
        if mode == "gray":
            for fidx, frame in enumerate(vis_r):
                f = plt.figure()
                plt.imshow(frame, cmap="gray")
                if fname:
                    frame = transform.resize(frame, (vid.shape[-2:]), order=3)
                    plt.imsave(fname.format(fidx),frame, cmap="gray")

        else:
            for fidx, frame in enumerate(vis_r):
                f, ax = plt.subplots(1,1)
                img = vid[:,fidx,...].transpose(1,2,0)

                frame = transform.resize(frame, (img.shape[:2]), order=3)
                if blur:
                    frame = filters.gaussian(frame, 0.02*max(img.shape[-2:]))
                frame = np.clip(frame,0, frame.max())
                frame -= frame.min()
                frame /= frame.max()
                if overlap:
                    cmap = plt.cm.Reds
                    t_cm = cmap
                    t_cm._init()
                    t_cm._lut[:, -1] = np.linspace(0, 0.8, 259)
                    frame_v = np.zeros_like(frame)
                    frame_v[frame > 0.1] = frame[frame > 0.1]
                    ax.imshow(img)
                    cb = ax.contourf(frame_v, cmap=t_cm)
                    plt.colorbar(cb)
                else:
                    cmap = plt.get_cmap("jet")
                    frame = cmap(frame)
                    plt.imshow(frame, interpolation="bicubic")
                plt.savefig(fname.format(fidx))
        return vis_r

class DeepTaylor(LRP):
    def fprop(self, layers, inp):
        inputs = []
        for layer in layers:
            inputs += [inp]
            inp = layer(inp)
        return inputs

    def zplusprop(self,mod, inp, R):
        # Get the positive weights
        w = mod.weight.data
        v = torch.clamp(w,min=0)
        if self.cuda:
            v = v.cuda()
        z = torch.mm(inp, v.t()) + 1e-9
        s = R / z
        c = torch.mm(s, v)
        R = inp * c
        return R

    def zbprop(self, mod, inp, R):
        w = mod.weight.data.cpu().numpy()
        v = np.maximum(0, w)
        u = np.minimum(0, w)
        v, u = torch.from_numpy(v), torch.from_numpy(u)
        if self.cuda:
            v, u = v.cuda(), u.cuda()

        x = inp
        l, h = -1 * inp, 1 * inp
        z = torch.mm(x,w) - torch.mm(l,v) - torch.mm(h,u) + 1e-9
        s = R / z
        R = x * torch.mm(s, w.T) - l * torch.mm(s, v.T) - h * torch.mm(s, u.T)
        return R

    def poolprop(self, mod, inp, R):
        x = torch.autograd.Variable(inp.data, requires_grad=True)
        z = mod(x) + 1e-9
        if len(R.shape) != len(z.shape):
            z = z.reshape_as(R)
        s = R / z
        z.backward(s)
        c = x.grad
        R = inp * c
        return R

    def convprop(self,mod, inp, R):
        """
        Takes advantage of the fact that zb and z+ rules can be expressed in terms of forward prop and grad prop
        """
        x = torch.autograd.Variable(inp.data, requires_grad=True)
        w = mod.weight.data.cpu().numpy()
        pself = copy.deepcopy(mod)
        pself.weight.data = torch.from_numpy(np.maximum(0, w))
        if self.cuda:
            pself = pself.cuda()
        z = pself(x) + 1e-9
        s = R / z
        z.backward(s)
        c = x.grad
        r = inp * c
        return r

    def firstconvprop(self, mod, inp, R):
        x = torch.autograd.Variable(inp.data, requires_grad=True)
        l = torch.autograd.Variable(0 + 0 * inp.data, requires_grad=True)
        h = torch.autograd.Variable(1 + 0 * inp.data, requires_grad=True)
        w = mod.weight.data.cpu().numpy()
        pself = copy.deepcopy(mod)
        iself = copy.deepcopy(mod)
        nself = copy.deepcopy(mod)

        pself.weight.data = torch.from_numpy(np.maximum(0, w))
        nself.weight.data = torch.from_numpy(np.minimum(0, w))

        if self.cuda:
            pself = pself.cuda()
            iself = iself.cuda()
            nself = nself.cuda()

        if mod.bias is not None:
            pself.bias = None
            nself.bias = None
            iself.bias = None

        i = iself(x)
        p = pself(l)
        n = nself(h)

        z = i - p - n + 1e-9
        s = R / z

        i.backward(s)
        p.backward(s)
        n.backward(s)

        R = x * x.grad - l * l.grad - h * h.grad
        return R

    def relprop(self, layers, inputs, R):
        first = False
        inputs = list(reversed(inputs))
        inputs = [inp.data for inp in inputs]
        layers = list(reversed(layers))
        for i, l in enumerate(layers):
            if i == len(layers) - 1:
                first = True
            inp = inputs[i]
            if type(l) is torch.nn.Conv3d:
                R = self.firstconvprop(l, inp, R) if first else self.convprop(l, inp, R)
            elif type(l) in [torch.nn.AvgPool3d, torch.nn.AdaptiveAvgPool3d, torch.nn.MaxPool3d]:
                R = self.poolprop(l, inp, R)
            elif type(l) is torch.nn.Linear:
                if first:
                    R = self.zbprop(l, inp, R)
                else:
                    R = self.zplusprop(l, inp, R)
        return R

class MWP(LRP):
    def fprop(self, layers, inp):
        inputs = []
        for layer in layers:
            inputs += [inp]
            inp = layer(inp)
        return inputs

    def getP(self, layer, inp, p_i, clamp):
        x = torch.autograd.Variable(inp, requires_grad=True)
        pself = copy.deepcopy(layer)
        if not isinstance(layer,
                          (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                           nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
                           nn.MaxPool3d)):
            # Get positive weights
            w = layer.weight.data
            w_p = clamp(w)
            pself.weight.data = w_p
        z = pself(x) + 1e-9
        # For flatten layers
        if len(p_i.shape) != len(z.shape):
            z = z.reshape_as(p_i)
        y = p_i / z
        z.backward(y)
        c = x.grad
        r = inp * c
        return r


    def relprop(self, layers, inputs, p_i):
        inputs = list(reversed(inputs))
        inputs = [inp.data for inp in inputs]
        layers = list(reversed(layers))
        p_j = p_i
        clamp = partial(torch.clamp, min=0)
        for input, layer in zip(inputs,layers):
            if isinstance(layer, (nn.Conv3d, nn.Conv2d, nn.Conv1d,
                                  nn.Linear,
                                  nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                                  nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
                                  nn.MaxPool3d)):
                p_j = self.getP(layer, input, p_j, clamp)
                if torch.isnan(p_j).any():
                    print(layer)
        return p_j


class cMWP(MWP):
    def relprop(self, layers, inputs, p_i):
        inputs = list(reversed(inputs))
        inputs = [inp.data for inp in inputs]
        layers = list(reversed(layers))
        p_j = p_i
        n_j = p_i
        pos = partial(torch.clamp, min=0)
        neg = partial(torch.clamp, max=0)
        for input, layer in zip(inputs,layers):
            if isinstance(layer, (nn.Conv3d, nn.Conv2d, nn.Conv1d,
                                  nn.Linear,
                                  nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                                  nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
                                  nn.MaxPool3d)):
                p_j = self.getP(layer, input, p_j, pos)
                n_j = self.getP(layer, input, n_j, neg)
        p_j -= n_j
        return p_j

