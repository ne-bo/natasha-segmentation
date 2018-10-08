import os
import math
import json
import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

from utils.util import ensure_dir


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, loss, metrics, resume, config, train_logger=None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.name = config['name']
        self.epochs = config['trainer']['epochs']
        print('self.epochs ', self.epochs)
        self.save_freq = config['trainer']['save_freq']
        self.verbosity = config['trainer']['verbosity']
        self.with_cuda = config['cuda'] and torch.cuda.is_available()
        if config['cuda'] and not torch.cuda.is_available():
            self.logger.warning('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
        else:
            self.gpu = torch.device('cuda:' + str(config['gpu']))
            self.model = self.model.to(self.gpu)

        self.train_logger = train_logger
        # here we add to the optimizer only those parameters that are not frozen!
        non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
        print('%d non_frozen_parameters ' % len(non_frozen_parameters))
        self.optimizer = getattr(optim, config['optimizer_type'])(
            non_frozen_parameters,
            **config['optimizer']
        )
        self.lr_scheduler = getattr(optim.lr_scheduler,
                                    config['lr_scheduler_type'], None)
        if self.lr_scheduler:
            self.lr_scheduler = self.lr_scheduler(self.optimizer, **config['lr_scheduler'])
            self.lr_scheduler_freq = config['lr_scheduler_freq']
        self.monitor = config['trainer']['monitor']
        self.monitor_mode = config['trainer']['monitor_mode']

        assert self.monitor_mode == 'min' or self.monitor_mode == 'max'
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf

        self.start_epoch = 1
        self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], self.name)
        ensure_dir(self.checkpoint_dir)
        json.dump(config, open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
                  indent=4, sort_keys=False)
        if resume:
            self._resume_checkpoint(resume)
            torch.cuda.empty_cache()
        print('self.monitor_best = ', self.monitor_best)

    def train(self):
        """
        Full training logic
        """
        print('self.start_epoch ', self.start_epoch)
        print('self.epochs + 1 ', self.epochs + 1)

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    for i, metric in enumerate(self.metrics):
                        log[metric.__name__] = result['metrics'][i]
                elif key == 'val_metrics':
                    for i, metric in enumerate(self.metrics):
                        log['val_' + metric.__name__] = result['val_metrics'][i]
                else:
                    log[key] = value
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    self.logger.info(log)
                    # for key, value in log.items():
                    #     self.logger.info('    {:15s}: {}'.format(str(key), value))

            if epoch == 200:
                print('self.monitor_best = ', self.monitor_best)
                self.monitor_best = 1.0
                print('self.monitor_best = ', self.monitor_best)

            if (self.monitor_mode == 'min' and log[self.monitor] < self.monitor_best) \
                    or (self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best):
                self.monitor_best = log[self.monitor]
                if epoch < 200:
                    self._save_checkpoint(epoch, log, save_best=True)
                else:
                    self._save_checkpoint(epoch, log, save_best=True, save_with_name=True)
            if epoch % self.save_freq == 0:
                self._save_checkpoint(epoch, log)
                torch.cuda.empty_cache()
            if self.lr_scheduler and epoch % self.lr_scheduler_freq == 0:
                if self.config['lr_scheduler_type'] == 'ReduceLROnPlateau':
                    self.lr_scheduler.step(log[self.monitor], epoch)
                else:
                    self.lr_scheduler.step(epoch)
                    lr = self.lr_scheduler.get_lr()[0]
                    self.logger.info('New Learning Rate: {:.6f}'.format(lr))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, log, save_best=False, save_with_name=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'logger': self.train_logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{:03d}-loss-{:.4f}.pth.tar'
                                .format(epoch, log['loss']))
        torch.save(state, filename)
        torch.cuda.empty_cache()
        if save_best:
            if save_with_name:
                os.rename(filename, os.path.join(self.checkpoint_dir, 'model_best_lovasz.pth.tar'))
                self.logger.info("Saving current best: {} ...".format('model_best_lovasz.pth.tar'))
            else:
                os.rename(filename, os.path.join(self.checkpoint_dir, 'model_best.pth.tar'))
                self.logger.info("Saving current best: {} ...".format('model_best.pth.tar'))

        else:
            self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        if self.start_epoch == 200:
            self.start_epoch = 1
        self.monitor_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.with_cuda:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self.gpu)
        self.train_logger = checkpoint['logger']
        self.config = checkpoint['config']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
        checkpoint = None
        torch.cuda.empty_cache()
