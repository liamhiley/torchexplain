import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, filters
from functools import partial
import copy
from models import layers
from collections import OrderedDict

class LRP:
    """
    Class for Layer-wise Relevance Propagation family of techniques.
    Works by finding the relevance of each layer in some way, and mapping it to the input for that layer, for each layer
    in reverse order in the model, ending with the relevance of the final layer propagated on to the input sample (image,
    signal, text)
    """
    def __init__(self, cuda):
        self.cuda = False

    def get_layers(self, mdl, sample, layer_list, inputs, outputs, hooks):
        """
        :param mdl: PyTorch module with all layers registered as nn.Module objs.
        :param inp: Optional input sample to forward pass through each layer and record.
        :return layers: List of all layers used in mdl
        """
        def save_inp(mod, in_, out_):
            layer_list.append(mod)
            print(mod)
            if len(in_) > 1:
                detached_input = torch.autograd.Variable(in_[1].data).requires_grad_()
                res = None

                for layer, inp in zip(reversed(layer_list[:-1]), reversed(inputs)):
                    if isinstance(inp, list):
                        x = sum(inp)
                    else:
                        x = inp
                    if isinstance(layer_list[-2], layers.Downsample):
                        x = layer_list[-2](x)
                    if x.shape == in_[0].shape:
                        if not (x - in_[0]).sum():
                            res = x
                        elif isinstance(inp, list):
                            x = sum([nn.ReLU()(t) for t in inp])
                            if not (x - in_[0]).sum():
                                res = x

                inputs.append([res,detached_input])
                # for inp in reversed(inputs):
            elif len(inputs) and len(inputs[-1]) > 1 and not isinstance(layer_list[-2], (nn.Conv3d, nn.AvgPool3d)):
                pself = copy.deepcopy(mod)
                pself._forward_hooks = OrderedDict([])
                inp = [pself(inp) for inp in inputs[-1]]
                inputs.append(inputs[-1])
            else:
                detached_input = torch.autograd.Variable(in_[0].data).requires_grad_()
                inputs.append(detached_input)


        children = list(mdl.named_children())
        if children:
            for name, child in children:
                self.get_layers(child, sample, layer_list, inputs, outputs, hooks)
        else:
            hooks.append(mdl.register_forward_hook(save_inp))

    def map(self, r, vid, **kwargs):
        if self.cuda:
            r = r.cpu()
        # vis_r = r.clamp(min=0).sum((0, 1)).data.numpy()
        vis_r = r.sum((0,1)).data.numpy()
        # vis_r /= abs(vis_r).max()
        # vis_r *= 0.5 #tone down 'peakiness'(?)
        # vis_r -= vis_r.min()
        # if vis_r.max() > 0:
        #     vis_r /= vis_r.max()
        mode=kwargs.pop("mode",[])
        fname = kwargs.pop("fname",[])
        vid = vid.cpu().numpy()
        blur = kwargs.pop("blur",[])
        overlap = kwargs.pop("overlap",[])
        offset = kwargs.pop("offset",0)
        if mode == "gray":
            for fidx, frame in enumerate(vis_r):
                f = plt.figure()
                plt.imshow(frame, cmap="gray")
                if fname:
                    frame = transform.resize(frame, (vid.shape[-2:]), order=3)
                    plt.imsave(fname.format(fidx),frame, cmap="gray")
                plt.close(f)
        else:
            for fidx, frame in enumerate(vis_r):
                f, ax = plt.subplots(1,1)
                img = vid[:,fidx,...].transpose(1,2,0)
                if img.max() > 1:
                    img /= 255

                frame = transform.resize(frame, (img.shape[:2]), order=3)
                if blur:
                    frame = filters.gaussian(frame, 0.02*max(img.shape[-2:]))
                if overlap:
                    cmap = plt.cm.Reds
                    t_cm = cmap
                    t_cm._init()
                    t_cm._lut[:, -1] = np.linspace(0, 0.8, 259)
                    frame_v = np.zeros_like(frame)
                    frame_v = frame
                    ax.set_axis_off()
                    ax.imshow(img)
                    cb = ax.contourf(frame_v, cmap="Reds", clim=[-1,1])
                    # plt.colorbar(cb)
                else:
                    cmap = plt.get_cmap("jet")
                    frame = cmap(frame)
                    plt.imshow(frame, interpolation="bicubic")
                plt.savefig(fname.format(fidx+(offset*16)))
                plt.close(f)
        return vis_r

class DeepTaylor(LRP):
    def __init__(self, cuda, l=-1, h=1):
        self.l = l
        self.h = h
        super(DeepTaylor,self).__init__(cuda)
    def zplusprop(self,mod, inp, R):
        # Get the positive weights
        w = mod.weight.data
        v = torch.clamp(w,min=0)
        if self.cuda:
            v = v.cuda()
        pself = copy.deepcopy(mod)
        pself.weight.data = v
        grad = None
        x = torch.Tensor(inp.data)
        def get_in_grad(mod, in_, out_):
            grad = in_
        pself.register_backward_hook(get_in_grad)
        if len(inp) > 1:
            res, x = inp
            x = res + x
        else:
            x = inp
        z = pself(x) + 1e-9
        s = R / z
        z.backward(s)
        c = inp.grad
        R = inp * c
        return R


    def zbprop(self, mod, inp, R):
        w = mod.weight.data.cpu().numpy()
        v = np.maximum(0, w)
        u = np.minimum(0, w)
        v, u = torch.from_numpy(v), torch.from_numpy(u)
        if self.cuda:
            v, u = v.cuda(), u.cuda()
        l, h = self.l * inp, self.h * inp
        z = torch.mm(inp,w) - torch.mm(l,v) - torch.mm(h,u) + 1e-9
        s = R / z
        R = inp * torch.mm(s, w.T) - l * torch.mm(s, v.T) - h * torch.mm(s, u.T)
        return R

    def poolprop(self, mod, inp, R):
        pself = copy.deepcopy(mod)
        if len(inp) > 1:
            res, x = inp
            z = pself(x + res) + 1e-9
        else:
            x = inp
            z = pself(x) + 1e-9
        if len(R.shape) != len(z.shape):
            z = z.reshape_as(R)
        s = R / z
        z.backward(s)
        c = x.grad
        R = x * c
        return R

    def convprop(self,mod, inp, R):
        """
        Takes advantage of the fact that zb and z+ rules can be expressed in terms of forward prop and grad prop
        """
        w = mod.weight.data.cpu().numpy()
        pself = copy.deepcopy(mod)
        pself.weight.data = torch.from_numpy(np.maximum(0, w))
        if self.cuda:
            pself = pself.cuda()
        if len(inp) > 1:
            res, x = inp
            z = pself(x + res) + 1e-9
        else:
            if inp.grad is not None:
                print("here")
            x = inp
            z = pself(x) + 1e-9
        s = R / z
        z.backward(s)
        c = x.grad
        r = x * c
        return r

    def firstconvprop(self, mod, inp, R):

        l = torch.autograd.Variable(self.l + 0 * inp.data, requires_grad=True)
        h = torch.autograd.Variable(self.h + 0 * inp.data, requires_grad=True)
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
        i = iself(inp)
        p = pself(l)
        n = nself(h)

        z = i - p - n + 1e-9
        s = R / z

        i.backward(s)
        p.backward(s)
        n.backward(s)

        R = inp * inp.grad - l * l.grad - h * h.grad
        return R

    def relprop(self, mdl, sample, R):
        first = False
        inputs = []
        outputs = []
        layer_list = []
        hooks = []
        self.get_layers(mdl,sample, layer_list, inputs, outputs, hooks)
        mdl(sample)
        for hook in hooks:
            hook.remove()
        inputs = list(reversed(inputs))
        layer_list = list(reversed(layer_list))
        # inputs = [inp.data for inp in inputs]
        for i, (in_,l) in enumerate(zip(inputs,layer_list)):
            if i == len(layer_list) - 1:
                first = True
            inp = inputs[i]
            if isinstance(inp, int):
                inp = inputs[inp]
            if type(l) is torch.nn.Conv3d:
                R = self.firstconvprop(l, inp, R) if first else self.convprop(l, inp, R)
            elif type(l) in [torch.nn.AvgPool3d, torch.nn.AdaptiveAvgPool3d, torch.nn.MaxPool3d]:
                R = self.poolprop(l, inp, R)
            elif isinstance(l, layers.Downsample) and l.mode == 'A':
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

    def getP(self, layer, x, p_i, clamp):
        pself = copy.deepcopy(layer)
        if not isinstance(layer,
                          (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                           nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
                           nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                           Downsample)):
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
        r = x * c
        return r


    def relprop(self, mdl, sample, p_i):
        inputs = []
        layers = []
        self.get_layers(mdl, sample, layers, inputs)
        mdl(sample)
        sample.requires_grad = True
        inputs = list(reversed(inputs))

        layers = list(reversed(layers))
        for in_ in inputs:
            in_.requires_grad_()
        p_j = p_i
        pos = partial(torch.clamp, min=0)
        for input, layer in zip(inputs, layers):
            if isinstance(layer, (nn.Conv3d, nn.Conv2d, nn.Conv1d,
                                  nn.Linear,
                                  nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                                  nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
                                  nn.MaxPool3d)):
                p_j = self.getP(layer, input, p_j, pos)
        return p_j


class cMWP(MWP):
    def relprop(self, mdl, sample, p_i):
        inputs = []
        layers = []
        self.get_layers(mdl,sample, layers, inputs)
        mdl(sample)
        sample.requires_grad = True
        inputs = list(reversed(inputs))
        layers = list(reversed(layers))
        for in_ in inputs:
            in_.requires_grad_()
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

