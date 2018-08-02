import csv
import os
import pickle

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F

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

        outputs_1 = network(images_themselves[:, :, :64,:64], depths)
        outputs_2 = network(images_themselves[:, :, 37:,:64], depths)
        outputs_3 = network(images_themselves[:, :, :64,37:], depths)
        outputs_4 = network(images_themselves[:, :, 37:,37:], depths)

        outputs = torch.from_numpy(np.zeros((outputs_1.shape[0], outputs_1.shape[1], 101, 101))).float().cuda()

        outputs[:, :, :64,:64] += outputs_1
        outputs[:, :, 37:,:64] += outputs_2
        outputs[:, :, :64,37:] += outputs_3
        outputs[:, :, 37:,37:] += outputs_4

        outputs[:, :, 37:64, :37] /= 2.0
        outputs[:, :, 37:64, 64:] /= 2.0
        outputs[:, :, :37, 37:64] /= 2.0
        outputs[:, :, 64:, 37:64] /= 2.0
        outputs[:, :, 37:64, 37:64] /= 4.0

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

    #
    # flattened_mask = np.zeros(predicted.shape).flatten()
    # for j in range(predicted.shape[2]):
    #     flattened_mask[predicted.shape[1] * j: predicted.shape[1] * (j + 1)] = predicted[:, :, j].squeeze(dim=0).cpu().numpy()
    # try_rle = rle_encode(flattened_mask)
    # # print('try_rle ', try_rle)
    # #input()
    # string_rle = ''
    # for number in try_rle:
    #     string_rle = string_rle + str(number) + ' '
    # string_rle = string_rle.strip()
    # # print('string_rle', string_rle)
    # return string_rle


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


