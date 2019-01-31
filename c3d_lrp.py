import cnn3d
import datasets
import deep_taylor
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import pickle
import PIL.Image as pim
from skimage.segmentation import mark_boundaries
import numpy as np
import cv2 as cv


cuda = False
if torch.cuda.is_available():
    cuda = True

# ucf101path = "/home/c1435690/Projects/DAIS-ITA/Development/Datasets/UCF-101/" # laptop
ucf101path = "/media/datasets/Video/UCF-101/" # dais-n



def explain(mdl, sample, class_map):
    # Get the inputs to each layer, for decomposition
    vid, lbl = sample
    if len(vid.shape) < 5:
        vid = torch.from_numpy(np.expand_dims(vid, 0))
    if type(vid) is np.ndarray:
        vid = torch.from_numpy(vid)
    if cuda:
        vid = vid.cuda()
    if type(mdl) is torch.nn.DataParallel:
        layers = mdl.module.get_layers(mdl.module, vid)
    else:
        layers = mdl.get_layers(mdl, vid)

    explnr = deep_taylor.DeepTaylorExplainer(cuda)

    input = explnr.fprop(layers, vid)

    out = mdl(vid)
    print("Predicted label {}".format(class_map[torch.topk(out,1)[1].item()]))
    true_out = torch.zeros_like(out)
    true_out[:,lbl - 1] = 1.0

    R = explnr.relprop(layers, input, out*true_out)

    R = R.data.cpu().numpy()

    frames = R.shape[-3]
    for f in range(frames):
        deep_taylor.visualise(R[:,:,f,...], deep_taylor.heatmap, "dt{}.png".format(f))
    return R


if __name__ == "__main__":
    train_cache_dir = '/media/datasets/Video/UCF-101/train_cache.bin'
    train_split_path = '/media/datasets/Video/UCF-101/trainlist01.txt'
    test_cache_dir = '/media/datasets/Video/UCF-101/val_cache.bin'
    test_split_path = '/media/datasets/Video/UCF-101/testlist01.txt'

    ds_path = ucf101path

    if os.path.exists(train_cache_dir):
       with open(train_cache_dir, "rb") as f:
           train_dict = pickle.load(f)
           class_map = train_dict["class_map"]
           train_cache = train_dict["cache"]
    else:
       print("Loaded training cache from " + ds_path + "train_cache.bin")
       class_map, train_cache = datasets.generate_cache_file(ds_path, train_split_path, train_cache_dir)

    if os.path.exists(test_cache_dir):
       with open(test_cache_dir, "rb") as f:
           test_dict = pickle.load(f)
           test_cache = test_dict["cache"]
    else:
       print("Loaded validation cache from " + test_cache_dir)
       _, test_cache = datasets.generate_cache_file(ds_path, test_split_path, test_cache_dir)

    ds_mean = train_cache[0][-2]
    ds_std = train_cache[0][-1]

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            # transforms.Normalize(ds_mean, ds_std)
        ]
    )

    test_ds = datasets.VideoDataset(112, 112, 16, test_cache, class_map, transform=transform, shuffle=True)

    sample= test_ds[1]

    display_vid = test_ds.unnorm_vid(0, ds_mean, ds_std)


    for f in range(16):
        frame = display_vid[f].transpose(1,2,0)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        name = "fr{}.png".format(f + 1)
        cv.imwrite(name, frame)

    layers = [3, 4, 6, 3]

    # block = cnn3d.Block
    # print("Building model")
    # net3d = cnn3d.RESNET3D(layers, block, 101, (112, 112))

    # print("Loading pre-trained weights")
    # weights_path = os.path.join(os.getcwd(), "..", "ucf101")
    # load_dict = torch.load(weights_path + "/2018-11-23.pt")

    print("Building model")
    net3d = cnn3d.vgg13()

    print("Loading pre-trained weights")
    weights_path = os.path.join(os.getcwd(), "..", "ucf101")
    load_dict = torch.load(weights_path + "/vgg13-2019-01-30-0.29846938775510207%.pt")

    if torch.cuda.device_count() > 1:
        net3d = torch.nn.DataParallel(net3d)

    net3d.load_state_dict(load_dict, strict=False)

    if cuda:
        net3d.cuda()
    print("Explaining video {}, with true label {}".format(train_cache[1][1], train_cache[1][2]))
    explanation = explain(net3d, sample, class_map)