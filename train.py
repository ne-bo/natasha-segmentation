import os
import json
import logging
import argparse
import torch
import jstyleson

from inference import inference
from model_natasha.model import *
from model_natasha.loss import *
from model_natasha.metric import *
from data_loader_natasha import SegmentationDataLoader
from trainer import Trainer
from logger import Logger

logging.basicConfig(level=logging.INFO, format='')


def main(config, resume):
    train_logger = Logger()

    loss = eval(config['loss'])

    model = NatashaSegmentation(config=config)

    print('Create train loader')
    train_data_loader = SegmentationDataLoader(config, name='train')

    if False:
        print('Create trainer')
        trainer = Trainer(
            model, loss,
            resume=resume,
            config=config,
            data_loader=train_data_loader,
            train_logger=train_logger,
            starting_checkpoint=None#'saved//NatashaSegmentation/seresnet152-epoch-148-loss-0.0072.pth.tar'
        )

        print('Start training')
        trainer.train()

    print('Create test loader')
    test_data_loader = SegmentationDataLoader(config, name='test')
    # checkpoint_for_model = torch.load('saved/NatashaSegmentation/model_best.pth.tar')
    checkpoint_for_model = torch.load('saved/NatashaSegmentation/model_best-resnet-34-bce-0.0065.pth.tar')
    model.load_state_dict(checkpoint_for_model['state_dict'])
    model.eval()

    # print('Do inference for train')
    # inference(train_data_loader, model)

    print('Do inference')
    inference(test_data_loader, model)


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        config = torch.load(args.resume)['config']
    elif args.config is not None:
        config = jstyleson.load(open(args.config))
        # path = os.path.join(config['trainer']['save_dir'], config['name'])
    assert config is not None

    main(config, args.resume)
