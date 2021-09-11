# torchexplain
Explainability functionality built on top of the PyTorch autograd package. Currently enables 2D and 3D CNN based networks to produce LRP-like [[1]](#1) explanations for a given input.

## Usage
### Input
Enable the autograd graph for your input tensor:
`input_sample.requires_grad_()`
### Model
The package redefines the most common layers you'd find in an NN (the full list is found in [[./explain/autograd.py]](autograd.py)). Define your model class using these (it should be sufficient to replace references to torch.nn with torchexplain).
### Explain
Get the output of your forward pass and explain using the tensor's builtin autograd backprop function. If you want to explain for a specific class, use the grad_mask to set all output neurons to 0 other than the target.
``
output = mdl(sample)
grad_mask = torch.zeros_like(output)
grad_mask[:,target_neuron] += 1
explanation = torch.grad.autograd(output, mdl, grad_mask)
``
## References
<a id="1">[1]</a> 
Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., and Samek, W. (2015).
On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation.
PLoS ONE 10:e0130140. doi: 10.1371/journal.pone.0130140
