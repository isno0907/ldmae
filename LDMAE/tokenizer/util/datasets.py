# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import json

from torchvision import datasets, transforms
import random

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class InatDataset(datasets.ImageFolder):
    def __init__(self, is_train,**kwargs):
        super().__init__(**kwargs)
        self.mode = 'train' if is_train else 'val'
        print('<Before>')
        print(self.samples[100])
        print(len(self.samples))
        
        annot_path = f'{self.root}/{self.mode}2019.json'
        with open(annot_path, 'r') as file:
            annot = json.load(file)
        self.samples = []
        for img, tgt in zip(annot['images'], annot['annotations']):
            self.samples.append([f"{self.root}/{img['file_name']}", tgt['category_id']])
        print('<After>')
        print(self.samples[100])
        print(len(self.samples))
            
    def __getitem__(self, index: int):
        
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample, target

class ImgDataset(datasets.ImageFolder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_retries = 10
        
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        failed = []
        for _ in range(self.num_retries):
            path, target = self.samples[index]
            try:
                sample = self.loader(path)
            except:
                try:
                    sample = self.loader(path) # one more time
                except:
                    failed.append(path)
                    index = random.randint(0, len(self.samples) - 1)
                    continue
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            return sample, target
        else:
            print('Failed to load {} after {} retries'.format(
                failed, self.num_retries
            ))
    
def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.dataset is not None:
        print(f'Downstream task with {args.dataset}')
        if args.dataset.lower() == 'inat':
            dataset = InatDataset(is_train = is_train, root=args.data_path, transform=transform)
        elif args.dataset.lower() == 'cifar100':
            from torchvision.datasets import CIFAR100
            dataset = CIFAR100(train = is_train, root=args.data_path, transform=transform)
        elif args.dataset.lower() == 'cub':
            from torchvision.datasets import ImageFolder
            if is_train:
                dataset = ImageFolder(root=args.data_path+'/train', transform=transform)
            else:
                dataset = ImageFolder(root=args.data_path+'/test', transform=transform)

    else:
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = ImgDataset(root=root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if args.dataset is not None:
            transform.transforms.insert(0, transforms.Lambda(lambda img: img.convert('RGB')))

        return transform

    # eval transform
    t = []
    if args.dataset is not None:
        t.append(transforms.Lambda(lambda image: image.convert("RGB")))
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
