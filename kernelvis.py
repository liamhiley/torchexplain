import torch
import torch.optim as optim
import torchvision

import copy
from functools import partial
import sys
class LayerVis():
    """
    Given a PyTorch model, generates a set of random inputs that maximally activate each neuron in the model.
    Accepted input shapes are 1D, 2D or 3D
    """
    def __init__(self, mdl, shape, **kwargs):
        self.mdl = mdl
        mdl.eval()
        self.shape = shape
        self.optimizer = self._get_optimizer(**kwargs)
        self.filter_out = {}

    def _get_optimizer(self, **kwargs):
        return partial(optim.SGD, lr=0.01, momentum=0.6)

    def create_hook(self, filter_idx):
        def get_output(mod, in_, out_):
            self.filter_out_[mod] = out_[filter_idx]
        return get_output

    def optimize_input(self, layer, filter_idx, input):
        """
        To single out a filter from a given layer, you can zero all the other weights in the layer, essentially
        turning off all other filters
        """

        # Record the output for the filter, and optimize on that

        hook = filter.register_forward_hook(self.get_output, filter_idx)
        return output, hook

    def run(self, neuron_list):
        named_children = list(self.mdl.named_children())
        for n, c in named_children:
            print(f"Now visualising filters in {n}")





if __name__ == '__main__':
    sys.path.append('/home/hileyl/scratch/Projects/PyTorch/pytorch-video-recognition/')
    from network import C3D_model
    import collections

    mdl = C3D_model.C3D(101)
    if hasattr(mdl, "module"):
        module = mdl.module
    else:
        module = mdl

    state = torch.load('save_20.pth')
    n_state = []
    for k, v in state['state_dict'].items():
        if 'module' in k:
            n_state.append((k[7:], v))
    if n_state == []:
        n_state = state['state_dict']
    else:
        n_state = collections.OrderedDict(n_state)
    mdl.load_state_dict(n_state)
    vis = LayerVis(mdl, (1,3,16,112,112))
    vis.run([0])