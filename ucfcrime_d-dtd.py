#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import torch
# from models import resnet, layers
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

paths = os.listdir("/media/datasets/Video/UCF_CRIME/Crime/Abuse/") + os.listdir("/media/datasets/Video/UCF_CRIME/Crime/Assault/") + os.listdir("/media/datasets/Video/UCF_CRIME/Crime/RoadAccidents/")

"""
UCFCRIME
"""
import cv2
import numpy as np
import torchvision as tv

temp_paths = []

for path in paths:
    if "Abuse" in path:
        vid_path = os.path.join("/media/datasets/Video/UCF_CRIME/Crime/Abuse/", path)
    elif "Assault" in path:
        vid_path = os.path.join("/media/datasets/Video/UCF_CRIME/Crime/Assault/", path)
    else:
        vid_path = os.path.join("/media/datasets/Video/UCF_CRIME/Crime/RoadAccidents/", path)
    rdr = cv2.VideoCapture(vid_path)
    length = int(rdr.get(cv2.CAP_PROP_FRAME_COUNT))
    split_path = []
    if length > 16:
        for segm in range(0,length,16):
            segm_path = (vid_path,[segm,min(segm+15,length-segm)])
            split_path.append(segm_path)
        vid_path = split_path
    else:
        vid_path = [vid_path]
    temp_paths += vid_path
    rdr.release()

paths = temp_paths

for path in tqdm(paths):
    if isinstance(path, tuple):
        start = path[1][0]
        rdr = cv2.VideoCapture(path[0])
    else:
        rdr = cv2.VideoCapture(path)
        start = 0
    h = rdr.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = rdr.get(cv2.CAP_PROP_FRAME_WIDTH)
    vid = np.zeros((1,3,16,int(h),int(w)))


    i = 0
    while i - start < 16:
        ret, f = rdr.read()
        if i < start:
            i+= 1
            continue
        if not ret:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        vid[0, :, i-start, ...] = f
        i += 1

    vid = torch.from_numpy(vid)
    sample = torch.zeros((1,3,16,112,112)).requires_grad_()
    for i in range(16):
        img = vid[0, ..., i, :, :].type(torch.uint8)
        img = tv.transforms.ToPILImage()(img)
        img = tv.transforms.Scale((112, 112))(img)
        img = tv.transforms.ToTensor()(img) * 255
        img = tv.transforms.Normalize(ds_mean, [1, 1, 1])(img)
        sample[0, ..., i, :, :] = img

    sample = sample.cuda()
    sample_out = mdl(sample)
    vid_out = sample_out#torch.cat(sample_out,0).mean(0, keepdim=True)
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
    import explain
    import matplotlib.pyplot as plt
    explain.epsilon = 0
    # print(explanation.shape)
    if not os.path.exists("d_dtd"):
        os.mkdir("d_dtd")
    # os.mkdir(f"d_dtd/Abuse")
    # print(f"Explaining evidence for {ds.class_names[vid_label]} in {vid_path}, with a softmax score of {vid_prob}")
    if isinstance(path, tuple):
        path = path[0]
    if "Abuse" in path:
        exp_path = f"d_dtd/Abuse"
    elif "Assault" in path:
        exp_path = f"d_dtd/Assault"
    else:
        exp_path = f"d_dtd/RoadAccidents"
    if not os.path.exists(exp_path.split("/")[-1]):
        os.mkdir(exp_path.split("/")[-1])
    # for idx, (sample, out) in enumerate(tqdm(zip(samples, sample_out))):
    explain.shortcut = None
    filter_out = torch.zeros_like(vid_out)
    filter_out[:, vid_label] = 1  # out[:,vid_label]
    pos_evidence = torch.autograd.grad(vid_out, sample, grad_outputs=filter_out, retain_graph=True)[0]
    pos_vis = pos_evidence.sum(dim=(0,1))
    pos_vis /= abs(pos_vis).max()
    for i in range(16):
        f = plt.figure()
        fig = plt.imshow(pos_vis[i,...].cpu(), cmap="seismic", clim=(-1,1))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(f"{exp_path}/dtd{start+i}.png", bbox_inches='tight', pad_inches = 0)
        # plt.show()
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
        plt.savefig(f"{exp_path}/spat{i+start}.png", bbox_inches='tight', pad_inches = 0)
        # plt.show()
        plt.close(f)
        f = plt.figure()
        fig = plt.imshow(temp_vis[i,...].cpu(), cmap="seismic", clim=(-1,1))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(f"{exp_path}/temp{i+start}.png", bbox_inches='tight', pad_inches = 0)
        # plt.show()
        plt.close(f)

    torch.save(pos_evidence, exp_path + "/relevance.pt")
    with open(exp_path + "/log.txt", "w") as f:
        f.write(f"Video: {vid_id}\nPredicted class: {ds.class_names[vid_label]}\nSoftmax score: {vid_prob}")

# In[ ]:




