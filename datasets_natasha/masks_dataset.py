import csv
import os
import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class ImagesWithMasksDataset(Dataset):
    def __init__(self, config, name):
        self.data_folder = config['data_loader']['data_dir_%s' % name]
        self.name = name
        self.config = config
        self.image_size = config['data_loader']['initial_image_size']
        if name == 'train':
            self.transform = transforms.Compose([
                # transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ])

        self.all_depths = {}
        self.images, self.masks, self.depths = self.get_images_with_masks()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if index % 10000 == 0:
            print('index ', index)
        image = self.transform(Image.open(self.images[index]).convert('RGB'))
        depth = torch.from_numpy(self.depths[index]).float().unsqueeze(dim=0)

        mask = torch.from_numpy(self.masks[index]).float().unsqueeze(dim=0)

        image_with_depth_and_mask = torch.cat((image, depth, mask), dim=0)

        if self.name == 'train':
            image_with_depth_and_mask = random_crop(image_with_depth_and_mask)

        image_with_depth = image_with_depth_and_mask[:3] # NATASHA actually removed depth info
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
                mask = row[1]
                depths[i] = self.all_depths[image_id] * depths[i]

                # if mask != '':
                #     mask = [int(number) for number in mask.split(' ')]
                #     assert len(mask) % 2 == 0, 'Mask should contain only pairs of numbers'
                #     for pair in range(len(mask) // 2):
                #         a = mask[pair * 2]
                #         b = mask[pair * 2 + 1]
                #         masks[i] = self.mark_pixels(masks[i], a, b)

                mask_from_image = np.array(Image.open(images[i].replace('images', 'masks')).convert('RGB'))[:, :, 1]/255.0
                masks[i] = mask_from_image
        return images, masks, depths

    def mark_pixels(self, mask_i, a, b):
        initial_shape = mask_i.shape
        mask_i = mask_i.reshape(-1, 1)
        mask_i[a:a + b] = 1.0
        mask_i = mask_i.reshape(initial_shape)
        return mask_i
        # for i in range(mask_i.shape[0]):
        #     for j in range(mask_i.shape[1]):
        #         current_index = j * mask_i.shape[0] + i + 1
        #         if a <= current_index < a + b:
        #             mask_i[i, j] = 1.0
        # return mask_i


def random_crop(image_with_depth_and_mask, w=64, h=64):
    full_width = image_with_depth_and_mask.shape[1]
    full_height = image_with_depth_and_mask.shape[2]
    assert w <= full_width
    assert h <= full_height
    x_start = np.random.randint(low=0, high=full_width - w)
    y_start = np.random.randint(low=0, high=full_height - h)
    return image_with_depth_and_mask[:, x_start:x_start + w, y_start:y_start + h]