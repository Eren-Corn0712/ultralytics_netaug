# Ultralytics YOLO 🚀, GPL-3.0 license
from datetime import datetime

import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.distributed as dist
from torch import Tensor
from pathlib import Path
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from ultralytics import YOLO
from ultralytics.yolo.utils import (LOGGER, ONLINE, RANK, ROOT, SETTINGS, TQDM_BAR_FORMAT, __version__,
                                    callbacks, colorstr, emojis, yaml_save, DEFAULT_CFG)
from ultralytics.yolo.utils.torch_utils import (smart_inference_mode)
from ultralytics.yolo.v8.detect import DetectionTrainer
from ultralytics.yolo.utils.torch_utils import de_parallel

from copy import deepcopy
from netaug.task import NetAugDetectionModel
from netaug.utils.reset_utils import AverageMeter
from torch.nn.modules.batchnorm import _BatchNorm


class NetAugTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        super().__init__(cfg, overrides)

        self.best_aug_model = self.wdir / 'aug_best.pt'
        self.last_aug_model = self.wdir / 'aug_last.pt'

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = NetAugDetectionModel(cfg, nc=self.data['nc'],
                                     verbose=verbose and RANK == -1,
                                     max_width=self.args.max_width,
                                     max_depth=self.args.max_depth,
                                     num_points=self.args.num_points)
        if weights:
            model.load(weights)
        return model

    def _do_train(self, world_size=1):
        if world_size > 1:
            self._setup_ddp(world_size)

        self._setup_train(world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100)  # number of warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            if self.args.sort_channel and epoch % 50 == 0 and epoch != 0:
                self.model.sort_channels()

            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    # Base model forward
                    self.model.set_base()  # Active channel set base!
                    preds = self.model(batch['img'])
                    self.loss, self.loss_items = self.criterion(preds, batch)

                    if epoch < self.args.stop_width_aug:
                        # Network Augmentation
                        self.model.set_aug()
                        preds = self.model(batch['img'])
                        loss_active, loss_active_items = self.criterion(preds, batch)
                        self.loss += loss_active
                        self.loss_items += loss_active_items

                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')

            if RANK in (-1, 0):

                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    # Reset batch norm
                    self.ema.ema.set_base()  # Active channel set base!
                    self.model.set_base()  # Active channel set base!
                    if self.args.reset_bn:
                        self.reset_batch_norm()
                    # The validate method will get ema model!
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')

    def save_model(self):
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model).export_module()).half(),
            'ema': deepcopy(self.ema.ema.export_module()).half(),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),  # save as dict
            'date': datetime.now().isoformat(),
            'version': __version__}

        # Save last, best and delete
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
        del ckpt

        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)).half(),
            'ema': deepcopy(self.ema.ema).half(),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),  # save as dict
            'date': datetime.now().isoformat(),
            'version': __version__}
        # Save last, best and delete
        torch.save(ckpt, self.last_aug_model)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best_aug_model)
        del ckpt

    @smart_inference_mode()
    def reset_batch_norm(self) -> None:
        # Set model and ema model to eval model and set base
        self.model.eval()
        self.ema.ema.eval()
        self.model.set_base()
        self.ema.ema.set_base()

        temp_model = deepcopy(self.ema.ema)

        bn_mean, bn_var = {}, {}
        for name, m in temp_model.named_modules():
            if isinstance(m, _BatchNorm):
                bn_mean[name] = AverageMeter()
                bn_var[name] = AverageMeter()

                def new_forward(batch_norm_layer, mean_estimate, var_estimate):
                    def lambda_forward(x):
                        x = x.contiguous()
                        batch_mean = (
                            x.mean(0, keepdim=True)
                            .mean(2, keepdim=True)
                            .mean(3, keepdim=True)
                        )  # 1, C, 1, 1
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = (
                            batch_var.mean(0, keepdim=True)
                            .mean(2, keepdim=True)
                            .mean(3, keepdim=True)
                        )  # 1, C, 1, 1

                        batch_mean = torch.squeeze(batch_mean).to(x.dtype)
                        batch_var = torch.squeeze(batch_var).to(x.dtype)

                        mean_estimate.update(batch_mean.data, x.size(0))
                        var_estimate.update(batch_var.data, x.size(0))

                        fea_dim = batch_mean.shape[0]

                        return F.batch_norm(
                            input=x,
                            running_mean=batch_mean,
                            running_var=batch_var,
                            weight=batch_norm_layer.weight[:fea_dim].to(x.dtype),
                            bias=batch_norm_layer.bias[:fea_dim].to(x.dtype),
                            training=False,
                            momentum=batch_norm_layer.momentum,
                            eps=batch_norm_layer.eps,
                        )

                    return lambda_forward

                m.forward = new_forward(m, bn_mean[name], bn_var[name])

        # skip if there is no batch normalization layers in the network
        if len(bn_mean) == 0:
            return

        pbar = enumerate(self.train_loader)
        # Update dataloader attributes (optional)
        LOGGER.info("%11s" % "Starting the Reset Batch norm!, Closing dataloader Data Augmentation")

        if hasattr(self.train_loader.dataset, 'augment'):
            self.train_loader.dataset.augment = False
            self.train_loader.dataset.transforms = self.train_loader.dataset.build_transforms(hyp=self.args)

        if RANK in (-1, 0):
            LOGGER.info(('\n' + '%11s' * 4) % (
                'Epoch', 'GPU_mem', 'Instances', 'Size'))
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), bar_format=TQDM_BAR_FORMAT)

        for i, batch in pbar:
            with torch.cuda.amp.autocast(self.amp):
                batch = self.preprocess_batch(batch)
                temp_model(batch['img'])  # we don't need the predict result.
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                pbar.set_description(
                    ('%11s' * 2 + '%11.4g' * 2) %
                    (f'{self.epoch + 1}/{self.epochs}', mem, batch['cls'].shape[0], batch['img'].shape[-1])
                )

        # Revise ema model
        for name, m in self.ema.ema.named_modules():
            if name in bn_mean and bn_mean[name].count > 0:
                feature_dim = bn_mean[name].avg.size(0)
                assert isinstance(m, _BatchNorm)
                m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg.clone().detach())
                m.running_var.data[:feature_dim].copy_(bn_var[name].avg.clone().detach())
        # Revise model
        for name, m in self.model.named_modules():
            if name in bn_mean and bn_mean[name].count > 0:
                feature_dim = bn_mean[name].avg.size(0)
                assert isinstance(m, _BatchNorm)
                m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg.clone().detach())
                m.running_var.data[:feature_dim].copy_(bn_var[name].avg.clone().detach())

        if RANK in (-1, 0):
            LOGGER.info("Reset Batch Norm Successful.")

        if hasattr(self.train_loader.dataset, 'augment'):
            self.train_loader.dataset.augment = True
            self.train_loader.dataset.transforms = self.train_loader.dataset.build_transforms(hyp=self.args)
