# Ultralytics YOLO 🚀, GPL-3.0 license
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

from ultralytics.yolo.v8.detect import DetectionTrainer
from ultralytics.yolo.v8.detect.train import Loss
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_yaml
from ultralytics.yolo.utils.torch_utils import de_parallel
from ultralytics.yolo.utils.tal import make_anchors

from kornia.losses import SSIMLoss
from kornia.enhance import normalize_min_max


class FeatureExtractor:
    def __init__(self, model, layers: str):
        self.model = model
        self.layers = layers
        self.features = {layer: torch.empty(0) for layer in layers}
        self.hooks = []

    def get_hooks(self):
        for layer_id in self.layers:
            layer = self.model._modules[layer_id]
            self.hooks.append(layer.register_forward_hook(self.save_outputs_hook(layer_id)))

    def reset_feature(self):
        self.features = {layer: torch.empty(0) for layer in self.layers}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def save_outputs_hook(self, layer_id: str):
        def fn(_, __, output):
            self.features[layer_id] = output

        return fn


class DistillationTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        super().__init__(cfg=cfg, overrides=overrides)
        self.teacher, _ = self._load_teacher_weights(ROOT / self.args.teacher_weights)

        self.kd_loss_names = []
        self.feat_loss_names = []

    def _load_teacher_weights(self, weights):
        suffix = Path(weights).suffix
        if suffix == '.pt':
            model, ckpt = attempt_load_one_weight(weights, self.device)
        else:
            weights = check_file(weights)
            model, ckpt = weights, None
        return model, ckpt

    def kd_criterion(self, tea_preds, stu_preds, batch):
        if not hasattr(self, 'compute_logit_kd'):
            self.compute_logit_kd = LogitKDLoss(de_parallel(self.model))
        return self.compute_logit_kd(tea_preds, stu_preds, batch)

    def feat_kd_criterion(self, tea_preds, stu_preds):
        if not hasattr(self, 'compute_feat_kd'):
            self.compute_feat_kd = StructuralKDLoss(de_parallel(self.model), self.adapt_ch)
            if hasattr(self, "scaler"):
                setattr(self.compute_feat_kd, "scaler", self.scaler)
        return self.compute_feat_kd(tea_preds, stu_preds)

    def progress_string(self):
        return ('\n' + '%11s' *
                (4 + len(self.loss_names) + len(self.kd_loss_names) + len(self.feat_loss_names))
                % (
                    'Epoch', 'GPU_mem', *self.loss_names, *self.kd_loss_names, *self.feat_loss_names, 'Instances',
                    'Size')
                )

    def _setup_teacher_train(self, world_size):
        self.teacher = self.teacher.to(self.device)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[RANK])

    def _setup_feature_extractor(self):
        self.extract = {
            'student': FeatureExtractor(de_parallel(self.model.model), self.args.layers),
            'teacher': FeatureExtractor(de_parallel(self.teacher.model), self.args.layers)
        }
        self.adapt_ch = {}
        device = next(self.model.parameters()).device
        dummy_input = torch.zeros((self.batch_size, 3, self.args.imgsz, self.args.imgsz), device=device)
        self.extract['student'].get_hooks(), self.extract['teacher'].get_hooks()
        self.model.eval(), self.teacher.eval()
        with torch.no_grad():
            _ = self.teacher(dummy_input)
            _ = self.model(dummy_input)
        self.extract['student'].remove_hooks(), self.extract['teacher'].remove_hooks()
        for (k1, v1), (k2, v2) in zip(self.extract['student'].features.items(),
                                      self.extract['teacher'].features.items()):
            assert k1 == k2, "Feature not equal."
            self.adapt_ch[k1] = [v1.shape[1], v2.shape[1]]
        LOGGER.info("Feature extractor already prepared!")
        s = ' '.join([f"{k.capitalize()} {v} \n" for k, v in self.adapt_ch.items()])
        LOGGER.info(f"Adaptive channel is")
        LOGGER.info(s)

    def _do_train(self, world_size=1):
        if world_size > 1:
            self._setup_ddp(world_size)

        self._setup_train(world_size)

        # Append kd loss name
        if self.args.logit_kd:
            for x in ("box", "cls", "dfl"):
                self.kd_loss_names.append(f"kd_{x}")
        if self.args.feat_kd:
            for x in range(len(self.args.layers)):
                self.kd_loss_names.append(f"feat_{x}")
            self._setup_feature_extractor()

        self._setup_teacher_train(world_size)

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
            self.teacher.eval()

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
                self.extract['student'].get_hooks(), self.extract['teacher'].get_hooks()
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    with torch.no_grad():
                        tea_preds = self.teacher(batch['img'])
                    preds = self.model(batch['img'])
                    if self.args.logit_kd:
                        self.loss, self.loss_items = self.kd_criterion(preds, batch, tea_preds)
                    else:
                        self.loss, self.loss_items = self.criterion(preds, batch)

                    if self.args.feat_kd:
                        tea_preds = list(self.extract['teacher'].features.values())
                        stu_preds = list(self.extract['student'].features.values())
                        self.feat_loss, self.feat_items = self.feat_kd_criterion(tea_preds, stu_preds)

                        self.loss += self.feat_loss
                        self.loss_items = torch.cat([self.loss_items, self.feat_items])

                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items

                self.extract['student'].remove_hooks(), self.extract['teacher'].remove_hooks()
                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni
                    if hasattr(self.compute_feat_kd, "optimizer"):
                        self.compute_feat_kd.optimizer_step()
                        # print(self.compute_feat_kd.adapter[0].weight.data[:3, :3, ...].squeeze())

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
                    self.loss_items = self.loss_items[:3]  # only box cls dfs loss
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


class KDLoss(nn.KLDivLoss):
    def __init__(self, temperature, alpha, beta, size_average=None, reduce=None, reduction='mean', log_target=False):
        """This code referenced to
                https://github.com/yoshitomo-matsubara/torchdistill/blob/main/torchdistill/losses/single.py
        """
        super().__init__(size_average, reduce, reduction, log_target)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1 - alpha if beta is None else beta
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, input: Tensor, target: Tensor, targets=None, *args, **kwargs) -> Tensor:
        soft_loss = super().forward(torch.log_softmax(input / self.temperature, dim=1),
                                    torch.softmax(target / self.temperature, dim=1))

        if self.alpha is None or self.alpha == 0 or targets is None:
            return soft_loss

        hard_loss = self.cross_entropy_loss(input, targets)
        return self.alpha * hard_loss + self.beta * (self.temperature ** 2) * soft_loss


class LogitKDLoss(Loss):
    def __init__(self, model):
        super().__init__(model)
        self.lkd = KDLoss(self.hyp.temperature, self.hyp.alpha, self.hyp.beta, reduction='none')

    def get_distri_scores(self, feats):
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
        return pred_distri, pred_scores

    @staticmethod
    def permute_contiguous(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.permute(0, 2, 1).contiguous()

    def __call__(self, preds, batch, tea_preds=None, *args, **kwargs):
        tea_feats, stu_feats = (p[1] if isinstance(p, tuple) else p for p in (tea_preds, preds))

        tea_pred_distri, tea_pred_scores = self.get_distri_scores(tea_feats)
        stu_pred_distri, stu_pred_scores = self.get_distri_scores(stu_feats)

        tea_pred_distri, tea_pred_scores, stu_pred_distri, stu_pred_scores = map(self.permute_contiguous, (
            tea_pred_distri, tea_pred_scores, stu_pred_distri, stu_pred_scores))

        dtype = stu_pred_scores.dtype
        batch_size = tea_pred_scores.shape[0]
        imgsz = torch.tensor(preds[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size
        anchor_points, stride_tensor = make_anchors(stu_feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        tea_pred_bboxes = self.bbox_decode(anchor_points, tea_pred_distri)  # xyxy, (b, h*w, 4)
        stu_pred_bboxes = self.bbox_decode(anchor_points, stu_pred_distri)  # xyxy, (b, h*w, 4)

        _, tea_target_bboxes, tea_target_scores, tea_fg_mask, _ = self.assigner(
            tea_pred_scores.detach().sigmoid(), (tea_pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        _, stu_target_bboxes, stu_target_scores, stu_fg_mask, _ = self.assigner(
            stu_pred_scores.detach().sigmoid(), (stu_pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        tea_target_bboxes /= stride_tensor
        tea_target_scores_sum = max(tea_target_scores.sum(), 1)

        stu_target_bboxes /= stride_tensor
        stu_target_scores_sum = max(stu_target_scores.sum(), 1)

        # Cls Loss
        loss = torch.zeros(3 + 3, device=self.device)  # box, cls, dfl + kd
        loss[1] = self.bce(stu_pred_scores, stu_target_scores.to(dtype)).sum() / stu_target_scores_sum  # BCE
        if self.hyp.kd_cls != 0:
            loss[4] = self.lkd(stu_pred_scores, tea_target_scores.to(dtype)).sum() / max(tea_target_scores_sum,
                                                                                         stu_target_scores_sum)  # Logit KD Loss
        # KD Cls Loss
        if stu_fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(stu_pred_distri, stu_pred_bboxes, anchor_points,
                                              stu_target_bboxes, stu_target_scores,
                                              stu_target_scores_sum, stu_fg_mask)

        if tea_fg_mask.sum() and self.hyp.kd_box != 0. and self.hyp.kd_dfl != 0.:
            loss[3], loss[5] = self.bbox_loss(stu_pred_distri, stu_pred_bboxes, anchor_points,
                                              tea_target_bboxes, tea_target_scores,
                                              tea_target_scores_sum, tea_fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.kd_box  # kd box gain
        loss[4] *= self.hyp.kd_cls  # kd cls gain
        loss[5] *= self.hyp.kd_dfl  # kd dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class StructuralKDLoss:
    def __init__(self, model, adapt_ch):  # model must be de-paralleled
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters
        m = model.model[-1]  # Detect() module

        self.hyp = h
        self.device = device
        self.ssim = SSIMLoss(window_size=11)
        self.adapter = nn.ModuleList([
            nn.Conv2d(v[0], v[1], kernel_size=1, stride=1) for k, v in adapt_ch.items()
        ])
        self.adapter.to(self.device)
        self.adapter.train()
        self.optimizer = optim.SGD(self.adapter.parameters(), lr=0.001, momentum=0.9)
        self.scaler = None

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def __call__(self, tea_preds, stu_preds):
        loss = torch.zeros(len(self.adapter), device=self.device)
        batch_size = tea_preds[0].shape[0]
        for idx, (tea_pred, stu_pred) in enumerate(zip(tea_preds, stu_preds)):
            stu_pred = self.adapter[idx](stu_pred)

            stu_pred = normalize_min_max(stu_pred)
            tea_pred = normalize_min_max(tea_pred)
            loss[idx] = self.ssim(stu_pred, tea_pred)

        loss *= self.hyp.kd_feat

        return loss.sum() * batch_size, loss.detach()
