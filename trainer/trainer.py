import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """

    def __init__(self, model, loss, resume, config,
                 data_loader, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics=None,
                                      resume=resume, config=config, train_logger=train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.log_step = int(np.sqrt(self.batch_size))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        """
        self.model.train()

        total_loss = 0
        for batch_idx, data in enumerate(self.data_loader):
            images, true_masks = data

            images = images.cuda()
            true_masks = true_masks.cuda()

            masks_pred = self.model(images)
            masks_probs = F.sigmoid(masks_pred)
            masks_probs_flat = masks_probs.view(-1)
            true_masks_flat = true_masks.view(-1)

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

        log = {
            'loss': total_loss / len(self.data_loader)
        }

        return log
