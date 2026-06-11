import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils import clip_gradient, AvgMeter
from torch.autograd import Variable
from datetime import datetime
import torch.nn.functional as F

import albumentations as A
from albumentations.core.composition import Compose, OneOf

from mmseg import __version__
from mmseg.models.segmentors import ColonFormer as UNet


def get_train_transform(trainsize: int) -> A.Compose:
    return A.Compose([
        A.Resize(trainsize, trainsize),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.5),
        A.GaussNoise(p=0.3),
        A.Equalize(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.5),
    ])


class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_paths, mask_paths, aug=True, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = cv2.resize(image, (352, 352))
            mask = cv2.resize(mask, (352, 352))

        image = image.astype('float32') / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:,:,np.newaxis]
        mask = mask.astype('float32') / 255
        mask = mask.transpose((2, 0, 1))

        return np.asarray(image), np.asarray(mask)

epsilon = 1e-7

def recall_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall

def precision_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision

def dice_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+epsilon))

def iou_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return recall*precision/(recall+precision-recall*precision + epsilon)


class FocalLossV1(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        logits = logits.float()
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wfocal = FocalLossV1()(pred, mask)
    wfocal = (wfocal*weit).sum(dim=(2,3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wfocal + wiou).mean()


def train(train_loader, model, optimizer, epoch, lr_scheduler, args):
    model.train()
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    dice, iou = AvgMeter(), AvgMeter()
    with torch.autograd.set_detect_anomaly(True):
        for i, pack in enumerate(train_loader, start=1):
            if epoch <= 1:
                optimizer.param_groups[0]["lr"] = (epoch * i) / (1.0 * total_step) * args.init_lr
            else:
                lr_scheduler.step()

            for rate in size_rates:
                optimizer.zero_grad()
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                trainsize = int(round(args.init_trainsize*rate/32)*32)
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                map4, map3, map2, map1 = model(images)
                map1 = F.upsample(map1, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                map2 = F.upsample(map2, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                map3 = F.upsample(map3, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                map4 = F.upsample(map4, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                loss = structure_loss(map1, gts) + structure_loss(map2, gts) + structure_loss(map3, gts) + structure_loss(map4, gts)
                dice_score = dice_m(map4, gts)
                iou_score = iou_m(map4, gts)
                loss.backward()
                clip_gradient(optimizer, args.clip)
                optimizer.step()
                if rate == 1:
                    loss_record.update(loss.data, args.batchsize)
                    dice.update(dice_score.data, args.batchsize)
                    iou.update(iou_score.data, args.batchsize)

            if i == total_step:
                print('{} Training Epoch [{:03d}/{:03d}], '
                        '[loss: {:0.4f}, dice: {:0.4f}, iou: {:0.4f}]'.
                        format(datetime.now(), epoch, args.num_epochs,
                                loss_record.show(), dice.show(), iou.show()))

    ckpt_path = save_path + 'last.pth'
    print('[Saving Checkpoint:]', ckpt_path)
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict()
    }
    torch.save(checkpoint, ckpt_path)


def evaluate(val_loader, model, args):
    model.eval()
    loss_record = AvgMeter()
    dice, iou = AvgMeter(), AvgMeter()
    with torch.no_grad():
        for images, gts in val_loader:
            images = images.cuda()
            gts = gts.cuda()
            map4, map3, map2, map1 = model(images)
            trainsize = args.init_trainsize
            map1 = F.upsample(map1, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            map2 = F.upsample(map2, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            map3 = F.upsample(map3, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            map4 = F.upsample(map4, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            loss = structure_loss(map1, gts) + structure_loss(map2, gts) + structure_loss(map3, gts) + structure_loss(map4, gts)
            dice_score = dice_m(map4, gts)
            iou_score = iou_m(map4, gts)
            loss_record.update(loss.data, 1)
            dice.update(dice_score.data, 1)
            iou.update(iou_score.data, 1)
    model.train()
    return loss_record.show(), dice.show(), iou.show()


def save_checkpoint(path, epoch, model, optimizer, lr_scheduler):
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict()
    }
    torch.save(checkpoint, path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--backbone', type=str, default='b3')
    parser.add_argument('--init_lr', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--init_trainsize', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--train_path', type=str, default='./data/TrainDataset')
    parser.add_argument('--val_path', type=str, default='./data/ValDataset')
    parser.add_argument('--train_save', type=str, default='ConlonFormerB3')
    parser.add_argument('--resume_path', type=str, default='')
    args = parser.parse_args()

    save_path = 'snapshots/{}/'.format(args.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    else:
        print("Save path existed")

    train_img_paths = sorted(glob('{}/images/*'.format(args.train_path)))
    train_mask_paths = sorted(glob('{}/masks/*'.format(args.train_path)))

    transform = get_train_transform(args.init_trainsize)
    train_dataset = Dataset(train_img_paths, train_mask_paths, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    val_img_paths = sorted(glob('{}/images/*'.format(args.val_path)))
    val_mask_paths = sorted(glob('{}/masks/*'.format(args.val_path)))
    val_transform = A.Compose([A.Resize(args.init_trainsize, args.init_trainsize)])
    val_dataset = Dataset(val_img_paths, val_mask_paths, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )

    total_step = len(train_loader)

    model = UNet(backbone=dict(
                    type='mit_{}'.format(args.backbone),
                    style='pytorch'),
                decode_head=dict(
                    type='UPerHead',
                    in_channels=[64, 128, 320, 512],
                    in_index=[0, 1, 2, 3],
                    channels=128,
                    dropout_ratio=0.1,
                    num_classes=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    align_corners=False,
                    decoder_params=dict(embed_dim=768),
                    loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
                neck=None,
                auxiliary_head=None,
                train_cfg=dict(),
                test_cfg=dict(mode='whole'),
                pretrained='pretrained/mit_{}.pth'.format(args.backbone)).cuda()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader)*args.num_epochs,
        eta_min=args.init_lr/1000,
    )

    start_epoch = 1
    if args.resume_path != '':
        checkpoint = torch.load(args.resume_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    print("#"*20, "Start Training", "#"*20)
    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.num_epochs+1):
        train(train_loader, model, optimizer, epoch, lr_scheduler, args)
        val_loss, val_dice, val_iou = evaluate(val_loader, model, args)
        print('{} Validation Epoch [{:03d}/{:03d}], [loss: {:0.4f}, dice: {:0.4f}, iou: {:0.4f}]'.
              format(datetime.now(), epoch, args.num_epochs, val_loss, val_dice, val_iou))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = save_path + 'best.pth'
            print('[Saving Best Checkpoint:]', ckpt_path)
            save_checkpoint(ckpt_path, epoch, model, optimizer, lr_scheduler)
