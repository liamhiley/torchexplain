#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import torch
from models import resnet, layers
sys.path.append('/home/hileyl/scratch/Projects/PyTorch/3D-ResNets-PyTorch')
from datasets import ucf101
import spatial_transforms, temporal_transforms
import lrp
from tqdm import tqdm
import os
import random

# In[2]:


norm_value = 1
# ds_mean = [114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value]
ds_mean = [90.0, 98.0, 102.0]
ds_std = [38.7568578 / norm_value, 37.88248729 / norm_value,40.02898126 / norm_value]
st = spatial_transforms.Compose([
    spatial_transforms.Scale(112),
    spatial_transforms.CornerCrop(112, 'c'),
    spatial_transforms.ToTensor(1),
    spatial_transforms.Normalize(ds_mean, [1,1,1]) #Hara et. al. normalised by mean only
    # spatial_transforms.Normalize(ds_mean, ds_std)
])
tt = temporal_transforms.LoopPadding(16)
ds = ucf101.UCF101(
    root_path="/media/datasets/Video/UCF-101/jpg",
    annotation_path="/media/datasets/Video/UCF-101/ucf101_01.json",
    subset="validation",
    n_samples_for_each_video=0,
    spatial_transform=st,
    temporal_transform=tt
)


# In[3]:


# shortcut_type="A"
# mdl = resnet.resnet34(num_classes=101, sample_size=112, sample_duration=16, shortcut_type=shortcut_type, range=(-max(ds_mean), 255-min(ds_mean)), dim=3)
# pretrain="/home/hileyl/scratch/Projects/PyTorch/3D-ResNets-PyTorch/results/resnet34_ucf101/save_15.pth"
# shortcut_type="B"
# mdl = resnet.resnet50(num_classes=101, sample_size=112, sample_duration=16, shortcut_type=shortcut_type, dim=3)
# pretrain="resnet50_ucf101_701.pth"
# state = torch.load(pretrain)
# sd = state["state_dict"]
# if torch.cuda.is_available():
#     mdl = torch.nn.DataParallel(mdl)  # Hara et. al. normalised by mean only
# mdl.load_state_dict(state_dict=sd, strict=False)
# first_lin = True
# if hasattr(mdl, "module"):
#     module = mdl.module
# else:
#     module = mdl

#         first_lin = False
from models import c3d
import collections
mdl = c3d.C3D(101, range=(-max(ds_mean), 255-min(ds_mean)))
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
module.load_state_dict(n_state)

# In[6]:

module.eval()
vid_id = None
samples = []
sample_out = []
mdl = mdl.cuda()
rdn = random.randint(0, len(ds))
for idx, data in enumerate(ds.data):
    if idx < rdn:
        continue
    if vid_id is None:
        vid_id = data["video_id"]
    if data["video_id"] != vid_id:
        break
    sample, target = ds[idx]
    sample.requires_grad_()
    if torch.cuda.is_available():
        sample = sample.cuda()
    sample = torch.unsqueeze(sample,0)
    out = module(sample)
    samples.append(sample)
    sample_out.append(out)
vid = torch.cat(samples, 2)
vid_out = torch.cat(sample_out,0).mean(0, keepdim=True)
fc, vid_label = vid_out.topk(1,1)
vid_label = vid_label.item()
vid_prob = torch.nn.Softmax()(vid_out)[:,vid_label]

# samp = torch.unsqueeze(samp, 0).cuda()
# print(samp.shape)
# out = mdl(samp)
# _, pred = out.topk(1,1)
# target["video_id"]


# In[5]:




explnr = lrp.DeepTaylor(True)
explanation = torch.zeros_like(vid)
import explain
import matplotlib.pyplot as plt
explain.epsilon = 0
# print(explanation.shape)
if not os.path.exists("d_dtd"):
    os.mkdir("d_dtd")
os.mkdir(f"d_dtd/{ds.class_names[vid_label]}")
print(f"Explaining evidence for {ds.class_names[vid_label]} in {vid_id}, with a softmax score of {vid_prob}")
exp_path = f"d_dtd/{ds.class_names[vid_label]}"
for idx, (sample, out) in enumerate(tqdm(zip(samples, sample_out))):
    explain.shortcut = None
    filter_out = torch.zeros_like(out)
    filter_out[:, vid_label] = 1  # out[:,vid_label]
    pos_evidence = torch.autograd.grad(out, sample, grad_outputs=filter_out, retain_graph=True)[0]
    pos_vis = pos_evidence.sum(dim=(0,1))
    pos_vis /= abs(pos_vis).max()
    for i in range(16):
        f = plt.figure()
        fig = plt.imshow(pos_vis[i,...].cpu(), cmap="seismic", clim=(-1,1))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(f"{exp_path}/dtd{(idx*16)+i}.png", bbox_inches='tight', pad_inches = 0)
        plt.show()
        plt.close(f)
    spatial_vis = torch.zeros_like(pos_vis)
    for f in range(16):
        frame = sample[:, :, f, ...]
        pad_frame = torch.zeros_like(sample).cuda()
        for i in range(16):
            pad_frame[:, :, i, ...] = frame
        pad_out = mdl(pad_frame)
        _, pad_label = pad_out.topk(1,1)
        pad_label = pad_label.item()
        print(ds.class_names[pad_label])
        pad_evidence = torch.autograd.grad(pad_out, pad_frame, grad_outputs=filter_out, retain_graph=True)[0]
        pad_vis = pad_evidence.sum(dim=(0,1))
        pad_vis /= abs(pad_vis).max()
        spatial_vis[f,...] = pad_vis[0]
    temp_vis = pos_vis - spatial_vis
    for i in range(16):
        f = plt.figure()
        fig = plt.imshow(spatial_vis[i,...].cpu(), cmap="seismic", clim=(-1,1))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(f"{exp_path}/spat{i+(idx*16)}.png", bbox_inches='tight', pad_inches = 0)
        plt.show()
        plt.close(f)
        f = plt.figure()
        fig = plt.imshow(temp_vis[i,...].cpu(), cmap="seismic", clim=(-1,1))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(f"{exp_path}/temp{i+(idx*16)}.png", bbox_inches='tight', pad_inches = 0)
        plt.show()
        plt.close(f)
    # explain.shortcut = "main"
    # main_evidence = torch.autograd.grad(out, sample, grad_outputs=filter_out, retain_graph=True)[0]
    # explain.shortcut = "shortcut"
    # sc_evidence = torch.autograd.grad(out, sample, grad_outputs=filter_out)[0]
    # pos_evidence = (pos_evidence - pos_evidence.min()) / (pos_evidence.max() - pos_evidence.min())
    # if sample.min() < 0:
    #     for c in range(3):
    #         sampleR[:,c,...] += ds_mean[c] / 255
    # sampleR = explnr.relprop(module, sample, filter_out*out)
    # display_sample = sample.detach()
    # for c in range(3):
    #     # display_sample[:, c, ...] *= ds_std[c]
    #     display_sample[:, c, ...] += ds_mean[c]


torch.save(pos_evidence, exp_path + "/relevance.pt")
with open(exp_path + "/log.txt", "w") as f:
    f.write(f"Video: {vid_id}\nPredicted class: {ds.class_names[vid_label]}\nSoftmax score: {vid_prob}")

# In[ ]:




