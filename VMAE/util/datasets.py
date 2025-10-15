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
import numpy as np
from torch.utils.data import Dataset
import random

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import os
from PIL import Image
import numpy as np
import torch
    
# Define the custom dataset
class CelebAMaskDataset(Dataset):
    def __init__(self, images_path, annot_path, target_classes, img_size=(256, 256), mask_size=(32, 32), return_downsized_image=False):
        self.images_path = images_path
        self.annot_path = annot_path
        self.target_classes = target_classes
        self.img_size = img_size
        self.mask_size = mask_size
        self.return_downsized_image = return_downsized_image
        
        # List all image files in the images_path
        self.image_files = sorted([f for f in os.listdir(images_path) if f.endswith('.jpg')])
        print(f"Total {len(self.image_files)} images.")

        self.annot_path_dict = {}
        # Loop through all subfolders
        for folder in os.listdir(annot_path):
            folder_path = os.path.join(annot_path, folder)
            
            # Ensure it's a directory
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    
                    # Store only if it's a file
                    if os.path.isfile(file_path):
                        self.annot_path_dict[filename] = file_path
        print(f"Total {len(self.annot_path_dict)} masks.")
                        
        # Define image transformations
        self.image_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ])
        if self.return_downsized_image:
            print('return image, mask, downsized image')
            self.image_transform_downsize = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(mask_size, interpolation=Image.NEAREST)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_path, img_filename)
        _image = Image.open(img_path).convert('RGB')  # Open image and convert to RGB
        
        image = self.image_transform(_image)  # Apply transformations to image
        if self.return_downsized_image:
            downsized_image = self.image_transform_downsize(_image)

        # Load and process masks
        base_filename = os.path.splitext(img_filename)[0]
        base_filename = str(base_filename).zfill(5)
        
        mask_dict = dict()
        
        for i, cls in enumerate(self.target_classes):
            mask_filename = f'{base_filename}_{cls}.png'
            mask_path = self.annot_path_dict.get(mask_filename)
            if mask_path is not None:
                mask = Image.open(mask_path).convert('L')  # Open mask as grayscale
                mask = self.mask_transform(mask)
                mask = torch.tensor(np.array(mask) > 0, dtype=torch.bool)
                if mask.sum() > 0:
                    mask_dict[cls] = mask
        
        if self.return_downsized_image:
            return image, mask_dict, downsized_image
        else:
            return image, mask_dict

def get_tiny_imagenet(full_dataset):
    class_list_file = '/data/projects/jeongwoo/tiny-imagenet/200_wnids.txt'

    # Load the 200 selected class names
    with open(class_list_file, 'r') as f:
        selected_classes = [line.strip() for line in f.readlines()]

    # Create a mapping from original class names to new labels (0–199)
    class_name_to_new_label = {class_name: idx for idx, class_name in enumerate(selected_classes)}

    # Filter samples and remap their labels
    filtered_samples = []
    for img_path, label in full_dataset.samples:
        class_name = os.path.basename(os.path.dirname(img_path))
        if class_name in selected_classes:
            new_label = class_name_to_new_label[class_name]
            filtered_samples.append((img_path, new_label))

    # Update the dataset with filtered samples and remapped labels
    full_dataset.samples = filtered_samples
    full_dataset.targets = [label for _, label in filtered_samples]

    return full_dataset


class ADE20KPatchDataset(Dataset):
    def __init__(self, root_dir, split='training', image_transform=None, annot_transform=None, return_downsized_image=False):
        print(f"ADE20K: {split}")
        self.image_transform = image_transform
        self.annot_transform = annot_transform
        
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.annotation_dir = os.path.join(root_dir, 'annotations', split)

        self.image_files = sorted(os.listdir(self.image_dir))
        self.annotation_files = sorted(os.listdir(self.annotation_dir))

        self.return_downsized_image = return_downsized_image
        if self.return_downsized_image:
            print('return image, mask, downsized image')
            self.image_transform_downsize = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ])
            
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        ann_path = os.path.join(self.annotation_dir, self.annotation_files[idx])

        _image = PIL.Image.open(img_path).convert("RGB")
        annotation = PIL.Image.open(ann_path)

        if self.image_transform:
            image = self.image_transform(_image)
        if self.annot_transform:
            annotation = self.annot_transform(annotation)
            
        if self.return_downsized_image:
            downsized_image = self.image_transform_downsize(_image)

        if self.return_downsized_image:
            return image, annotation, downsized_image
        else:
            return image, annotation
    
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
