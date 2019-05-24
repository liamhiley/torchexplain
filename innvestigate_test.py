#!/usr/bin/env python
# coding: utf-8

# In[3]:


import innvestigate
import innvestigate.utils
import keras.applications.vgg16 as vgg16

# Get TF model
tf_mdl, preprocess = vgg16.VGG16(), vgg16.preprocess_input
tf_mdl.load_weights("weights.h5")
# Strip softmax layer
tf_mdl = innvestigate.utils.model_wo_softmax(tf_mdl)
#Get weights as list
tf_weights = tf_mdl.get_weights()


# In[4]:


from models import vgg
import torch
import explain
# Get Torch model
mean = [103.939, 116.779, 123.68]
py_mdl = vgg.vgg16(range=(-max(mean), 255-min(mean)))

tf_i = 0
# Convert weights over from TF to Torch
for ft in py_mdl.features:
    if hasattr(ft, "weight"):
        w, b = tf_weights[tf_i:tf_i+2]
        w = torch.tensor(w.transpose(3,2,0,1))
        b = torch.tensor(b)
        ft.weight.data = w
        ft.bias.data = b
        tf_i += 2
for ft in py_mdl.classifier:
    if hasattr(ft, "weight"):
        w, b = tf_weights[tf_i:tf_i+2]
        w = torch.tensor(w.transpose(1,0))
        b = torch.tensor(b)
        ft.weight.data = w
        ft.bias.data = b
        tf_i += 2
print("Loaded weights")


# In[5]:


import cv2, numpy as np, matplotlib.pyplot as plt
# Load image
image = cv2.imread("readme_example_input.png")
# Resize for VGG
image = cv2.resize(image, dsize=(224,224))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()


# In[6]:


# Preprocess image and duplicate for both models
tf_in = preprocess(image[None])
py_in = torch.from_numpy(np.flip(tf_in,axis=0).copy()).permute(0,3,1,2).requires_grad_()


# In[2]:


py_out = py_mdl(py_in)
v, i = py_out.topk(1,1)
i = i.item()
filter_out = torch.zeros_like(py_out)
filter_out[:, i] += 1
pos_evidence = torch.autograd.grad(py_out, py_in, grad_outputs=filter_out)


# In[1]:


print(pos_evidence)


# In[ ]:




