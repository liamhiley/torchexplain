import torch
import torch.nn as nn
import copy
from functools import partial
import matplotlib.pyplot as plt
from skimage import transform, filters
import numpy as np
import cv2

class MWP:
    def __init__(self, cuda):
        self.cuda = True

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
        p_j = inp * c
        return p_j


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

    def map(self, p_j, **kwargs):
        if self.cuda:
            p_j = p_j.cpu()
        vis_p = p_j.clamp(min=0).sum((0, 1)).data.numpy()
        mode=kwargs.pop("mode",[])
        fname = kwargs.pop("fname",[])
        vid = kwargs.pop("vid",[])[[2,1,0],...]
        blur = kwargs.pop("blur",[])
        overlap = kwargs.pop("overlap",[])
        if mode == "gray":
            for fidx, frame in enumerate(vis_p):
                f = plt.figure()
                plt.imshow(frame, cmap="gray")
                if fname:
                    frame = transform.resize(frame, (vid.shape[-2:]), order=3)
                    plt.imsave(fname.format(fidx),frame, cmap="gray")

        else:
            for fidx, frame in enumerate(vis_p):
                f, ax = plt.subplots(1,1)
                img = vid[:,fidx,...].transpose(1,2,0)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
        return vis_p


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
