import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import cv2
import numpy as np
import random
import scipy.ndimage as ndi
from tqdm import tqdm
import os
from PIL import Image
from skimage.io import imread
from skimage import measure


def scale_radius(src, img_size, padding=False):
    """Normalize image based on retinal radius"""
    x = src[src.shape[0] // 2, ...].sum(axis=1)
    r = (x > x.mean() / 10).sum() // 2
    yx = src.sum(axis=2)
    region_props = measure.regionprops((yx > yx.mean() / 10).astype('uint8'))
    yc, xc = np.round(region_props[0].centroid).astype('int')
    x1 = max(xc - r, 0)
    x2 = min(xc + r, src.shape[1] - 1)
    y1 = max(yc - r, 0)
    y2 = min(yc + r, src.shape[0] - 1)
    dst = src[y1:y2, x1:x2]
    dst = cv2.resize(dst, dsize=None, fx=img_size/(2*r), fy=img_size/(2*r))
    if padding:
        pad_x = (img_size - dst.shape[1]) // 2
        pad_y = (img_size - dst.shape[0]) // 2
        dst = np.pad(dst, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 'constant')
    return dst


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform=None, scale_radius_enabled=True, img_size=288):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.scale_radius_enabled = scale_radius_enabled
        self.img_size = img_size

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        
        # Apply scale_radius preprocessing on-the-fly if enabled
        if self.scale_radius_enabled:
            try:
                img = scale_radius(img, img_size=self.img_size, padding=False)
            except Exception as e:
                # If scale_radius fails, just resize
                img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


    def __len__(self):
        return len(self.img_paths)
