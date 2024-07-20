import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io

class SHHA_New(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root

        prefix = "train_data" if train else "test_data"
        self.prefix = prefix

        self.img_list = os.listdir(f"{data_root}/{prefix}/images")

        # 获取gts
        self.gt_list = {}
        for img_name in self.img_list:
            img_path = f"{data_root}/{prefix}/images/{img_name}"
            gt_path = f"{data_root}/{prefix}/ground-truth/GT_{img_name}"

            # points
            self.gt_list[img_path] = {}
            self.gt_list[img_path]['points'] = gt_path.replace("jpg", "mat")

            # seg_map
            img_id = img_name.split('_')[1].split('.')[0]
            self.gt_list[img_path]['seg_map'] = os.path.join(data_root, prefix, f'pmap', f'PMAP_{img_id}.mat')

        self.img_list = sorted(list(self.gt_list.keys()))
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        gt_path = self.gt_list[img_path]['points']
        gt_seg_path = self.gt_list[img_path]['seg_map']

        # load image and ground truth
        img, img_seg, points = load_data((img_path, gt_path, gt_seg_path), self.train)
        # applu augumentation
        if self.transform is not None:
            img = self.transform(img)

        img_seg = torch.from_numpy(img_seg).unsqueeze(0)
        if self.train:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            # scale the image and points
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                img_seg = torch.nn.functional.upsample_bilinear(img_seg.unsqueeze(0), scale_factor=scale).squeeze(0)
                points *= scale

        # random crop augumentaiton
        if self.train and self.patch:
            img, img_seg, points = random_crop(img, img_seg, points)
            for i, _ in enumerate(points):
                points[i] = torch.Tensor(points[i])

        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            img_seg = torch.Tensor(img_seg[:, :, :, ::-1].copy())
            for i, _ in enumerate(points):
                points[i][:, 0] = 128 - points[i][:, 0]

        if not self.train:
            points = [points]

        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for i in range(len(points))]
        for i, _ in enumerate(points):
            target[i]['point'] = torch.Tensor(points[i])
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([points[i].shape[0]]).long()
            target[i]['seg_map'] =torch.Tensor(img_seg[i])

        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path, gt_seg_path = img_gt_path

    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # image segmentation map
    img_seg = io.loadmat(gt_seg_path)['pmap']

    points = io.loadmat(gt_path)['image_info'][0][0][0][0][0]
    if train:
        points = np.unique(points, axis=0)

    return img, img_seg, np.array(points)

# random crop augumentation
def random_crop(img, img_seg, den, num_patch=4):
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_img_seg = np.zeros([num_patch, img_seg.shape[0], half_h, half_w])
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        result_img_seg[i] = img_seg[:, start_h:end_h, start_w:end_w]
        # copy the cropped points
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        # shift the corrdinates
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        result_den.append(record_den)

    return result_img, result_img_seg, result_den