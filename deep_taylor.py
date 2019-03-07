import torch
import numpy as np
import copy
import cv2 as cv


class DeepTaylorExplainer:
    def __init__(self, cuda = False):
        self.cuda = cuda




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
    R = R.cpu().data.numpy()
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