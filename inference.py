import csv
import os
import pickle

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F

from datasets_natasha.masks_dataset import resize_image
from utils.crf import dense_crf
from utils.util import rle_encode, RLenc


def outputs_for_large_dataset(loader, network):
    config = loader.config
    torch.cuda.empty_cache()
    name = loader.dataset.name
    batches_number = save_inference_results_on_disk(loader, network, name)
    # batches_number = 45
    name = 'test'
    return read_inference_results_from_disk(config, batches_number, name)


def read_inference_results_from_disk(config, batches_number, name):
    path = os.path.join(config['temp_folder'], name, '')
    pack_volume = config['pack_volume']
    assert 'all_outputs_%d' % pack_volume in os.listdir(path), \
        'There should be precomputed inference data in %s!' % path

    all_outputs = torch.cuda.FloatTensor()
    for i in range(1, batches_number + 1):
        outputs = torch.load('%sall_outputs_%d' % (path, i * pack_volume))
        all_outputs = torch.cat((all_outputs, outputs), dim=0)

    return all_outputs


def save_inference_results_on_disk(loader, network, name):
    config = loader.config
    pack_volume = config['pack_volume']
    path = os.path.join(config['temp_folder'], name, '')
    print('path ', path)
    network.eval()

    all_outputs = torch.cuda.FloatTensor()
    i = 1
    print('Inference is in progress')
    print('loader ', loader.batch_sampler.sampler)
    for data in tqdm(loader):
        images, true_masks = data

        images = images.cuda()

        images_themselves = images[:, :3]
        if config['with_depth']:
            depths = images[:, 3]
        else:
            depths = None

        size_101 = config['101']

        if config['resize_128']:
            outputs = network(images_themselves, depths)
        else:
            size_patch = config['patch_size']
            size_37 = size_101 - size_patch
            outputs_1 = network(images_themselves[:, :, :size_patch,:size_patch], depths)
            outputs_2 = network(images_themselves[:, :, size_37:,:size_patch], depths)
            outputs_3 = network(images_themselves[:, :, :size_patch,size_37:], depths)
            outputs_4 = network(images_themselves[:, :, size_37:,size_37:], depths)

            outputs = torch.from_numpy(np.zeros((outputs_1.shape[0], outputs_1.shape[1], size_101, size_101))).float().cuda()

            outputs[:, :, :size_patch,:size_patch] += outputs_1
            outputs[:, :, size_37:,:size_patch] += outputs_2
            outputs[:, :, :size_patch,size_37:] += outputs_3
            outputs[:, :, size_37:,size_37:] += outputs_4

            outputs[:, :, size_37:size_patch, :size_37] /= 2.0
            outputs[:, :, size_37:size_patch, size_patch:] /= 2.0
            outputs[:, :, :size_37, size_37:size_patch] /= 2.0
            outputs[:, :, size_patch:, size_37:size_patch] /= 2.0
            outputs[:, :, size_37:size_patch, size_37:size_patch] /= 4.0

        outputs = F.sigmoid(outputs)

        # something like smoothing with conditional random fields
        if config['crf']:
            for j, (output, image) in enumerate(zip(outputs, images_themselves)):
                output = output.squeeze(dim=0)
                # print('output before ', output.shape)
                image = torch.transpose(image, dim0=0, dim1=2)
                # print('image ', image.shape)
                output = dense_crf(image.data.cpu().numpy().astype(np.uint8), output.data.cpu().numpy())
                # print('output after', output)
                outputs[j] = torch.from_numpy(output).float()

        if config['resize_128']:
            resized_outputs = np.zeros((outputs.shape[0], outputs.shape[1], size_101, size_101))
            for j, output in enumerate(outputs):
                resized_outputs[j] = resize_image(output.data.cpu().numpy(), (size_101, size_101))
            outputs = torch.from_numpy(resized_outputs).cuda().float()

        all_outputs = torch.cat((all_outputs, outputs.data), dim=0)

        if i % pack_volume == 0:
            torch.save(all_outputs, '%sall_outputs_%d' % (path, i))
            all_outputs = torch.cuda.FloatTensor()
            torch.cuda.empty_cache()
        i += 1
    batches_number = len(loader) // pack_volume
    print('batches_number = ', batches_number)
    all_outputs = None
    torch.cuda.empty_cache()
    return batches_number


def convert_output_to_rle(output):
    predicted = output.gt(0.2)
    rle = RLenc(predicted[0].cpu().numpy(), order='F', format=True)
    return rle


def inference(loader, model):
    config = loader.config

    all_outputs = outputs_for_large_dataset(loader, model)

    all_ids = []
    with open(config['data_loader']['sample_submission_csv'], 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', dialect='excel')
        rows = list(reader)
        for i, row in tqdm(enumerate(rows[1:])):
            all_ids.append(row[0])

    rles = []
    for id, output in zip(all_ids, all_outputs):
        rles.append(convert_output_to_rle(output))
        print('id ', id)

    rows = []
    with open('natasha_submission.csv', 'w') as csv_file:
        csv_file.write('id,rle_mask\n')
        for (id, rle) in zip(all_ids, rles):
            row = str(id) + ',' + rle + '\n'
            rows.append(row)
        rows[-1] = rows[-1].replace('\n', '')
        csv_file.writelines(rows)


