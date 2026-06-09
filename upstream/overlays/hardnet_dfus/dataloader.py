# albumentations
import json
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import os
import glob
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.utils import square_padding
from pathlib import Path
from PIL import Image


_SPLIT_FILE = Path(__file__).resolve().parents[4] / "data" / "splits" / "kvasir_split.json"


def calculatemns(img_list, size, rect):
    mean = 0.
    std = 0.
    for name in img_list:
        image = Image.open(name).convert('RGB')
        w, h = image.size
        if rect:
            image = square_padding(image, w, h)
        image = transforms.Resize((size, size))(transforms.ToTensor()(image))
        image = image.flatten(1)
        mean += image.mean(1)
        std += image.std(1)
    mean /= len(img_list)
    std /= len(img_list)
    return mean, std


def split_data(length, ratio, k=0, seed=42, k_fold=1):
    """Return train/val/test indices from the shared kvasir_split.json."""
    with open(_SPLIT_FILE) as f:
        split = json.load(f)
    train_idx = split["train"]
    val_idx = split["val"]
    test_idx = split["test"]
    print(len(train_idx), " ", len(val_idx), " ", len(test_idx))
    return train_idx, val_idx, test_idx


class create_dataset(data.Dataset):
    def __init__(self, data_path, trainsize, augmentations, train=True, train_ratio=0.8, rect=False, k=0, k_fold=1, seed=None):
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.ratio = train_ratio
        self.rect = rect
        try:
            f = []
            for p in data_path if isinstance(data_path, list) else [data_path]:
                p = Path(p)
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)

            self.images = sorted([x for x in f if ('images' in x) and (x.endswith('.jpg') or x.endswith('.png'))])
            self.gts = sorted([x for x in f if ('masks' in x) and (x.endswith('.jpg') or x.endswith('.png'))])
            length = len(self.images)

            train_idx, val_idx, test_idx = split_data(length, self.ratio, k=k, seed=seed, k_fold=k_fold)

            if train:
                self.images = sorted([self.images[idx] for idx in train_idx])
                self.gts = sorted([self.gts[idx] for idx in train_idx])
                for i in range(len(self.images)):
                    assert self.images[i].split(os.sep)[-1].split('.')[0] == self.gts[i].split(os.sep)[-1].split('.')[0]
                print('load %g training data from %g images in %s' % (len(self.images), length, data_path))
            else:
                self.images = sorted([self.images[idx] for idx in val_idx])
                self.gts = sorted([self.gts[idx] for idx in val_idx])
                for i in range(len(self.images)):
                    assert self.images[i].split(os.sep)[-1].split('.')[0] == self.gts[i].split(os.sep)[-1].split('.')[0]
                print('load %g validation data from %g images in %s' % (len(self.images), length, data_path))

        except Exception as e:
            raise Exception('Error loading data from %s: %s\n' % (data_path, e))

        self.size = len(self.images)
        if self.augmentations:
            print("data augmentation (standardized)")
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5),
                A.Perspective(scale=(0.05, 0.1), p=0.5),
                A.GaussNoise(p=0.3),
                A.Equalize(p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.5),
            ])
        else:
            print("no data augmentation")
            self.transform = None
        self.nom = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.totensor = A.Compose([ToTensorV2()])

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(self.gts[index], cv2.IMREAD_GRAYSCALE)
        name = self.gts[index]

        if self.rect:
            if image.shape[0] > image.shape[1]:
                total = A.PadIfNeeded(p=1, min_height=image.shape[0], min_width=image.shape[0], border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)(image=image, mask=gt)
                image = total['image']
                gt = total['mask']
                assert image.shape[0] == image.shape[1], '1, %s, %g/%g' % (self.images[index], image.shape[0], image.shape[1])
            elif image.shape[0] < image.shape[1]:
                total = A.PadIfNeeded(p=1, min_height=image.shape[1], min_width=image.shape[1], border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)(image=image, mask=gt)
                image = total['image']
                gt = total['mask']
                assert image.shape[0] == image.shape[1], '2, %s, %g/%g' % (self.images[index], image.shape[0], image.shape[1])

        if self.augmentations and self.transform is not None:
            total = self.transform(image=image, mask=gt)
            image = total["image"]
            gt = total['mask']

        total = A.Resize(self.trainsize, self.trainsize)(image=image, mask=gt)
        image = total["image"]
        gt = total['mask']

        image_final = self.totensor(image=image)
        image = image_final["image"]
        image = self.nom(image)

        gt_final = self.totensor(image=total["mask"], mask=total["mask"])
        gt = gt_final["mask"]
        return image, gt.unsqueeze(0), name

    def __len__(self):
        return self.size


class test_dataset(data.Dataset):
    def __init__(self, data_path, size, rect):
        self.trainsize = size
        try:
            f = []
            for p in data_path if isinstance(data_path, list) else [data_path]:
                p = Path(p)
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            self.images = sorted([x for x in f if 'image' in x])
            length = len(self.images)
        except Exception as e:
            raise Exception('Error loading data from %s: %s\n' % (data_path, e))

        print('load %g all images' % length, 'from', data_path)
        self.rect = rect
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        name = self.images[index]
        image = Image.open(name).convert('RGB')
        image0 = np.array(image)
        w, h = image.size
        if self.rect:
            image = square_padding(image, w, h)
        image = self.transform(image)
        return image.unsqueeze(0), name, (h, w), image0

    def __len__(self):
        return self.size
