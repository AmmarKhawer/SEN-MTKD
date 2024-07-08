import os 
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms
import random
import torch
from PIL import  Image



class Dess2(Dataset):
    def __init__(self, train_path, mask_path, images_list, masks_list, transform=None):
        self.images_list = images_list
        self.masks_list = masks_list
        self.train_path = train_path
        self.mask_path = mask_path
        self.transform = transform

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.train_path, self.images_list[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        mask = cv2.imread(os.path.join(self.mask_path, self.masks_list[index]), cv2.IMREAD_GRAYSCALE)

        # Apply transformation if specified
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask, self.images_list[index]

    def __len__(self):
        return len(self.images_list)


class dess400(Dataset):
    def __init__(self, train_path, mask_path, indices, transform=None):
        self.train_path = train_path
        self.mask_path = mask_path
        self.n_samples = len(indices)
        self.transform = transform
        self.indices = indices

    def __getitem__(self, index):
        name = self.indices[index]

        image = cv2.imread(os.path.join(self.train_path, name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        mask = cv2.imread(os.path.join(self.mask_path, name), cv2.IMREAD_GRAYSCALE)

        # Apply transformation if specified
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask, name

    def __len__(self):
        return self.n_samples


import albumentations as A
from albumentations.pytorch import ToTensorV2
import re



def splitset(ids,factor):
    split_index = int(len(ids) * factor)
    train= ids[:split_index]
    test = ids[split_index:]
    return train ,test

def extract_numerical_part(element):
    match = re.search(r'\d+', element)
    if match:
        return int(match.group())
    else:
        return 0


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


class Mix(Dataset):
    def __init__(self, train_path, mask_path, testB, test_gtB, mix_images, transform=None):
        self.mix_images = mix_images
        self.train_path = train_path
        self.mask_path = mask_path
        self.b_path = testB
        self.b_mask_path = test_gtB

        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4814, 0.4494, 0.3958],
                                 std=[0.2563, 0.2516, 0.2601])]
        )
        self.noise_transform = transforms.Compose([

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0))]
        )
        self.mask_transform = transforms.Compose([
            transforms.Resize(384),

            transforms.PILToTensor()
        ])

    def __getitem__(self, index):

        full_path = self.mix_images[index]
        image = Image.open(full_path).convert("RGB")

        image_ema = self.noise_transform(image)
        image = self.transform(image)

        if full_path.split("/")[-2] == "train":
            mask = Image.open(os.path.join(self.mask_path, full_path.split("/")[-1])).convert("L")
            mask = self.mask_transform(mask)
            return image, image_ema, mask, mask
        else:
            mask = Image.open(os.path.join(self.b_mask_path, full_path.split("/")[-1])).convert("L")
            mask = self.mask_transform(mask)
            empty_mask = torch.full((1, 384, 384), -1, dtype=torch.int64)

            return image, image, empty_mask, mask

    def __len__(self):
        return len(self.mix_images)


class hotencoded(Dataset):
    def __init__(self, train_path, mask_path, images_list, masks_list, num_classes, transform=None):
        self.images_list = images_list
        self.masks_list = masks_list
        self.train_path = train_path
        self.mask_path = mask_path
        self.transform = transform
        self.num_classes = num_classes

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.train_path, self.images_list[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        mask = cv2.imread(os.path.join(self.mask_path, self.masks_list[index]), cv2.IMREAD_GRAYSCALE)

        # Apply transformation if specified
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Perform one-hot encoding on the mask
        one_hot_mask = torch.zeros((self.num_classes, *mask.shape), dtype=torch.float32)

        # Expand dimensions of `mask` to match the shape of `one_hot_mask`
        expanded_mask = mask.unsqueeze(0)

        # Convert `expanded_mask` to `torch.int64`
        expanded_mask = expanded_mask.to(torch.int64)

        # Use `torch.scatter_` to perform the one-hot encoding
        one_hot_mask = one_hot_mask.scatter_(0, expanded_mask, 1)

        return image, one_hot_mask, mask, self.images_list[index]

    def __len__(self):
        return len(self.images_list)


class hot400(Dataset):
    def __init__(self, train_path, mask_path, indices, num_classes, transform=None):
        self.train_path = train_path
        self.mask_path = mask_path
        self.n_samples = len(indices)
        self.transform = transform
        self.indices = indices
        self.num_classes = num_classes

    def __getitem__(self, index):
        name = self.indices[index]

        image = cv2.imread(os.path.join(self.train_path, name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        mask = cv2.imread(os.path.join(self.mask_path, name), cv2.IMREAD_GRAYSCALE)

        # Apply transformation if specified
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        # Perform one-hot encoding on the mask
        one_hot_mask = torch.zeros((self.num_classes, *mask.shape), dtype=torch.float32)

        # Expand dimensions of `mask` to match the shape of `one_hot_mask`
        expanded_mask = mask.unsqueeze(0)

        # Convert `expanded_mask` to `torch.int64`
        expanded_mask = expanded_mask.to(torch.int64)

        # Use `torch.scatter_` to perform the one-hot encoding
        one_hot_mask = one_hot_mask.scatter_(0, expanded_mask, 1)

        return image, one_hot_mask, mask, name

    def __len__(self):
        return self.n_samples

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
    A.PadIfNeeded(min_height=384, min_width=384, always_apply=True, border_mode=cv2.BORDER_REPLICATE),
    # A.RandomCrop(height=320, width=320, always_apply=True),
    A.GaussNoise(p=0.2),
    A.Perspective(p=0.5),
    A.OneOf(
        [
            A.Sharpen(p=1),
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ],
        p=0.9,
    ),

    ToTensorV2(),  # Convert image and mask to tensors
])
# A.Lambda(mask=round_clip_0_1),

valid_transform = A.Compose([
    A.Resize(384, 384),
    # Add normalization if required
    ToTensorV2()  # Convert image to tensor
])



