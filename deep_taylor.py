import torch
import numpy as np
import copy
import cv2 as cv


class DeepTaylorExplainer:
    def __init__(self, cuda = False):
        self.cuda = cuda

    def fprop(self, layers, inp):
        inputs = []
        for layer in layers:
            inputs += [inp]
            inp = layer(inp)
        return inputs

    def zplusprop(self,mod, inp, R):
        # Get the positive weights
        w = mod.weight.data
        v = np.maximum(0, w)
        if self.cuda:
            v = v.cuda()
        z = torch.mm(inp, v.t()) + 1e-9
        s = R / z
        c = torch.mm(s, v)
        R = inp * c
        return R

    def zbprop(self, mod, inp, R):
        w = mod.weight.data
        v, u = np.maximum(0, w), np.minimum(0, w)
        if self.cuda:
            v, u = v.cuda(), u.cuda()

        x = inp
        l, h = -1 * inp, 1 * inp
        z = x @ w - l @ v - h @ u + 1e-9
        s = R / z
        R = x * (s @ w.T) - l * (s @ v.T) - h * (s @ u.T)
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
        w = mod.weight.data
        pself = copy.deepcopy(mod)
        pself.weight.data = np.maximum(0, w)
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
        l = torch.autograd.Variable(-1 + 0 * inp.data, requires_grad=True)
        h = torch.autograd.Variable(1 + 0 * inp.data, requires_grad=True)
        w = mod.weight.data
        pself = copy.deepcopy(mod)
        iself = copy.deepcopy(mod)
        nself = copy.deepcopy(mod)

        pself.weight.data = np.maximum(0, w)
        nself.weight.data = np.minimum(0, w)

        if self.cuda:
            pself = pself.cuda()
            iself = iself.cuda()
            nself = nself.cuda()

        if mod.bias:
            pself.bias *= 0
            nself.bias *= 0
            iself.bias *= 0

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


def heatmap(x):

	x = x[...,np.newaxis]

	# positive relevance
	hrp = 0.9 - np.clip(x-0.3,0,0.7)/0.7*0.5
	hgp = 0.9 - np.clip(x-0.0,0,0.3)/0.3*0.5 - np.clip(x-0.3,0,0.7)/0.7*0.4
	hbp = 0.9 - np.clip(x-0.0,0,0.3)/0.3*0.5 - np.clip(x-0.3,0,0.7)/0.7*0.4

	# negative relevance
	hrn = 0.9 - np.clip(-x-0.0,0,0.3)/0.3*0.5 - np.clip(-x-0.3,0,0.7)/0.7*0.4
	hgn = 0.9 - np.clip(-x-0.0,0,0.3)/0.3*0.5 - np.clip(-x-0.3,0,0.7)/0.7*0.4
	hbn = 0.9 - np.clip(-x-0.3,0,0.7)/0.7*0.5

	r = hrp*(x>=0)+hrn*(x<0)
	g = hgp*(x>=0)+hgn*(x<0)
	b = hbp*(x>=0)+hbn*(x<0)

	return np.concatenate([r,g,b],axis=-1)

def visualise(R, colormap, name):
    N = len(R)
    assert (N <= 16)
    R = R[0].transpose(1,2,0)
    R = np.sum(R, axis=2)
    R = colormap(R / np.abs(R).max())
    # Create a mosaic and upsample
    R = R.reshape([1, N, 112, 112, 3])
    R = np.pad(R, ((0, 0), (0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=1)
    R = R.transpose([0, 2, 1, 3, 4]).reshape([116, N * 116, 3])
    R = np.kron(R, np.ones([2, 2, 1]))
    R *= 255
    R = cv.cvtColor(R.astype(np.uint8), cv.COLOR_RGB2BGR)
    cv.imwrite(name, R)
    return R