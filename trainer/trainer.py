import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

from base import BaseTrainer
from model_natasha.loss import lovasz


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """

    def __init__(self, model, loss, resume, config,
                 data_loader, train_logger=None, starting_checkpoint=None):
        super(Trainer, self).__init__(model, loss, metrics=None,
                                      resume=resume, config=config, train_logger=train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.log_step = int(np.sqrt(self.batch_size))
        # self.loss_after_100 = eval(config['loss_after_100'])
        if starting_checkpoint:
            self._resume_checkpoint(starting_checkpoint)
        print('self.monitor_best = ', self.monitor_best)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        """
        if epoch == 200:
            for g in self.optimizer.param_groups:
                g['lr'] = 0.0001
                # print('g ', g)
            self.lr_scheduler = getattr(torch.optim.lr_scheduler,
                                        self.config['lr_scheduler_type'], None)
            self.lr_scheduler = self.lr_scheduler(self.optimizer, **self.config['lr_scheduler'])
            self.lr_scheduler_freq = self.config['lr_scheduler_freq']
            print('self.lr_scheduler ', self.lr_scheduler.get_lr())
        torch.cuda.empty_cache()
        self.model.train()

        total_loss = 0
        for batch_idx, data in enumerate(self.data_loader):
            torch.cuda.empty_cache()
            images, true_masks = data

            images = images.cuda()
            true_masks = true_masks.cuda()

            images_themselves = images[:, :3]
            if self.config['with_depth']:
               depths = images[:, 3]
            else:
                depths = None

            masks_pred = self.model(images_themselves, depths)
            masks_probs = F.sigmoid(masks_pred)
            masks_probs_flat = masks_probs.view(-1)
            true_masks_flat = true_masks.view(-1)

            if epoch >= 200: # self.config['loss'] == 'lovasz':
                loss = lovasz(masks_pred, true_masks)
            else:
                 loss = self.loss(masks_probs_flat, true_masks_flat)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    len(self.data_loader) * self.data_loader.batch_size,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
        # print('chaeck aug')
        # input()
        torch.cuda.empty_cache()
        log = {
            'loss': total_loss / len(self.data_loader)
        }

        return log
