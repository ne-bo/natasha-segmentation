import csv
import os
import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.transform import resize
from torch.utils.data import Dataset
from tqdm import tqdm

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)


def strong_aug(p=.5, config=None):
    return Compose([
        HorizontalFlip(),
        # OneOf([
        #     IAAAdditiveGaussianNoise(),
        #     GaussNoise(),
        # ], p=0.2),
        # OneOf([
        #     MotionBlur(p=.2),
        #     MedianBlur(blur_limit=3, p=.1),
        #     Blur(blur_limit=3, p=.1),
        # ], p=0.2),
        ShiftScaleRotate(shift_limit=0.001, scale_limit=0.1, rotate_limit=10, p=.2),
        # Compose([
        #     OpticalDistortion(p=0.3),
        #     GridDistortion(p=.1),
        #     IAAPiecewiseAffine(p=0.3),
        # ], p=0.2),
        # OneOf([
        #     CLAHE(clip_limit=2),
        #     IAASharpen(),
        #     IAAEmboss(),
        #     RandomContrast(),
        #     RandomBrightness(),
        # ], p=0.3),
        # HueSaturationValue(p=0.3),
    ], p=p)


def tta_aug(p=.5, config=None):
    return Compose([
        # HorizontalFlip(),
        # OneOf([
        #     IAAAdditiveGaussianNoise(),
        #     GaussNoise(),
        # ], p=0.2),
        # OneOf([
        #     MotionBlur(p=.2),
        #     MedianBlur(blur_limit=3, p=.1),
        #     Blur(blur_limit=3, p=.1),
        # ], p=0.2),
        # ShiftScaleRotate(shift_limit=0.001, scale_limit=0.1, rotate_limit=10, p=.2),
        # Compose([
        #     OpticalDistortion(p=0.3),
        #     GridDistortion(p=.1),
        #     IAAPiecewiseAffine(p=0.3),
        # ], p=0.2),
        # OneOf([
        #     CLAHE(clip_limit=2),
        #     IAASharpen(),
        #     IAAEmboss(),
        #     RandomContrast(),
        #     RandomBrightness(),
        # ], p=0.3),
        # HueSaturationValue(p=0.3),
    ], p=p)


class ImagesWithMasksDataset(Dataset):
    def __init__(self, config, name):
        self.data_folder = config['data_loader']['data_dir_%s' % name]
        self.name = name
        self.config = config
        if self.config['resize_128']:
            self.image_size = 128
        else:
            self.image_size = config['data_loader']['initial_image_size']
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        self.all_depths = {}
        self.images, self.masks, self.depths = self.get_images_with_masks()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if index % 10000 == 0:
            print('index ', index)

        image = Image.open(self.images[index]).convert('RGB')

        if self.config['resize_128']:
            pad_up_to_128 = transforms.Pad(padding=(14, 27, 13, 0), padding_mode='reflect')
            # pad_up_to_128 = transforms.Pad(padding=(14, 14, 13, 13), padding_mode='reflect')
            image = pad_up_to_128(image)

        image = np.array(image)
        depth = self.depths[index]
        mask = self.masks[index]

        whatever_data = "my name"
        if self.name == 'train':
            augmentation = strong_aug(p=0.5, config=self.config)
        else:
            augmentation = tta_aug(p=0.5, config=self.config)

        data = {"image": image, "mask": mask, "whatever_data": whatever_data, "additional": "hello"}
        augmented = augmentation(**data)
        image, mask, whatever_data, additional = augmented["image"], augmented["mask"], \
                                                 augmented["whatever_data"], \
                                                 augmented["additional"]

        image = transforms.ToTensor()(image).float()
        mask = torch.from_numpy(mask).float().unsqueeze(dim=0)
        depth = torch.from_numpy(depth).float().unsqueeze(dim=0)
        image_with_depth_and_mask = torch.cat((image, depth, mask), dim=0)

        # if self.name == 'train':
        #     image_with_depth_and_mask = random_crop(image_with_depth_and_mask)
        # image_with_depth_and_mask = flip_h_w(image_with_depth_and_mask, direction='horizontal')
            # we should not use vertical flips because ofthe data'snature
            # https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/61998
            # image_with_depth_and_mask = flip_h_w(image_with_depth_and_mask, direction='vertical')

        image_with_depth = image_with_depth_and_mask[:4]
        mask = image_with_depth_and_mask[-1]
        # print('image_with_depth ', image_with_depth.shape)
        # print('self.masks[index] ', mask, mask.shape)
        return image_with_depth, mask

    def get_images_with_masks(self):

        with open(self.config['data_loader']['depths_csv'], 'r') as csv_file_depth:
            reader_depth = csv.reader(csv_file_depth, delimiter=',', dialect='excel')
            rows_depth = list(reader_depth)
            for row in rows_depth[1:]:
                image_id = row[0]
                depth = float(row[1])
                self.all_depths[image_id] = depth

        if self.name == 'train':
            images, masks, depths = self.get_train_images_from_csv()
        else:
            images, masks, depths = self.get_test_images_from_csv()

        images = np.array(images)
        return images, masks, depths

    def get_test_images_from_csv(self):
        images = []
        with open(self.config['data_loader']['sample_submission_csv'], 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',', dialect='excel')
            rows = list(reader)
            for i, row in tqdm(enumerate(rows[1:])):
                images.append(row[0] + '.png')
        print('test images ', images)
        depths = np.ones((len(images), self.image_size, self.image_size), dtype=float)
        for i, image in enumerate(images):
            image_id = image.replace(self.config['data_loader']['data_dir_test'], '').replace('.png', '')
            images[i] = os.path.join(self.config['data_loader']['data_dir_test'], image)
            depths[i] = self.all_depths[image_id] * depths[i]
        return images, np.zeros((len(images), self.image_size, self.image_size)), depths

    def get_train_images_from_csv(self):
        images = os.listdir(self.config['data_loader']['data_dir_train'])
        print('train images ', images)

        masks = np.zeros((len(images), self.image_size, self.image_size))
        depths = np.ones((len(images), self.image_size, self.image_size))
        images = []

        with open(self.config['data_loader']['train_masks_csv'], 'r') as csv_file_mask:
            reader_mask = csv.reader(csv_file_mask, delimiter=',', dialect='excel')
            rows_mask = list(reader_mask)
            for i, row in tqdm(enumerate(rows_mask[1:])):
                image_id = row[0]
                images.append(os.path.join(self.config['data_loader']['data_dir_train'], image_id + '.png'))
                depths[i] = self.all_depths[image_id] * depths[i]

                mask_as_image = Image.open(images[i].replace('images', 'masks')).convert('RGB')
                if self.config['resize_128']:
                    pad_up_to_128 = transforms.Pad(padding=(14, 27, 13, 0), padding_mode='reflect')
                    mask_as_image = pad_up_to_128(mask_as_image)
                # if self.config['resize_128']:
                #     mask_as_image = mask_as_image.resize((128, 128))

                mask_from_image = np.array(mask_as_image)[:, :, 1] / 255.0

                masks[i] = mask_from_image
        return images, masks, depths


def random_crop(image_with_depth_and_mask, w=64, h=64):
    full_width = image_with_depth_and_mask.shape[1]
    full_height = image_with_depth_and_mask.shape[2]
    assert w <= full_width
    assert h <= full_height
    x_start = np.random.randint(low=0, high=full_width - w)
    y_start = np.random.randint(low=0, high=full_height - h)
    return image_with_depth_and_mask[:, x_start:x_start + w, y_start:y_start + h]


# https://github.com/pytorch/pytorch/issues/229
def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def flip_h_w(image_with_depth_and_mask, direction='horizontal'):
    coin = np.random.randint(low=0, high=100)
    if coin > 50:
        if direction == 'horizontal':
            image_with_depth_and_mask = flip(image_with_depth_and_mask, dim=2)
        if direction == 'vertical':
            image_with_depth_and_mask = flip(image_with_depth_and_mask, dim=1)

    return image_with_depth_and_mask


def resize_image(image, target_size):
    """Resize image to target size

    Args:
        image (numpy.ndarray): Image of shape (C x H x W).
        target_size (tuple): Target size (H, W).

    Returns:
        numpy.ndarray: Resized image of shape (C x H x W).

    """
    n_channels = image.shape[0]
    resized_image = resize(image, (n_channels, target_size[0], target_size[1]), mode='constant')
    return resized_image
