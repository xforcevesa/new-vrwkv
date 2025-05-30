# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
import torch.distributed
from augmentations import collate_data_and_cast_aug
from datasets import build_dataset

from losses_hint import DistillationLoss
from samplers import RASampler
from functools import partial

import importlib
import utils
import random
import math
from multiprocessing import Value
from abc import ABC

import sys
from typing import Iterable, Optional
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import utils


class MaskingGenerator(ABC):
    def __init__(self, input_size):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width

    def __repr__(self):
        raise NotImplementedError

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        raise NotImplementedError

    def get_none_mask(self):
        return np.zeros(shape=self.get_shape(), dtype=bool)
    
    
    
class RandomMaskingGenerator(MaskingGenerator):
    def __init__(
        self,
        input_size,
    ):
        """
        Args:
            input_size: the size of the token map, e.g., 14x14
        """
        super().__init__(input_size)

    def __repr__(self):
        repr_str = f"Random Generator({self.height}, {self.width})"
        return repr_str

    def _mask(self, mask, max_mask_patches):
        return super()._mask(mask, max_mask_patches)

    def __call__(self, num_masking_patches=0):
        if num_masking_patches <= 0:
            return np.zeros(shape=self.get_shape(), dtype=bool)

        mask = np.hstack([np.ones(num_masking_patches, dtype=bool),
                          np.zeros(self.num_patches - num_masking_patches, dtype=bool)])
        np.random.shuffle(mask)
        mask = mask.reshape(self.get_shape())
        return mask


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str)
    parser.add_argument('--target_model', default='deit_base_patch16_224', type=str)
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # add dataset parameters
    parser.add_argument('--global_crops_size', '--img_size', default=224, type=int,
                        help="this should be equal to image size")
    parser.add_argument('--patch_size', default=16, type=int,
                        help="patch size for vit patch embedding")
    
    # add masking parameter
    parser.add_argument('--mask_ratio', default=(0.1, 0.5), type=float, nargs='+',
                        help="mask ratio can be either a value or a range")
    parser.add_argument('--mask_probability', default=0., type=float,
                        help="how many samples with be applied with masking")
    parser.add_argument('--mask_first_n', action='store_true',
                        help="mask the first n sample to avoid shuffling. Needed for MAE-style encoder")
    parser.add_argument('--clone_batch', default=1, type=int,
                        help="how many times to clone the batch for masking (default: 1, not cloning)")
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='base', type=str)
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    parser.add_argument('--lambda_token', type=float, default=1.0)
    parser.add_argument('--lambda_fea', type=float, default=1.0)
    parser.add_argument('--lambda_patch', type=float, default=1.0)
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true')
    parser.add_argument('--weight_inherit', default='') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'IMNET_ibot', 'IMNET_ibot_aug', 'IMNET_ibot_fast_aug', 'INAT', 'INAT19', 'IMNET_L', 'IMNET_L_ibot'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool=True):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()

def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            sampler_val = RASampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(dataset_val)
        
    n_tokens = (args.global_crops_size // args.patch_size) ** 2
    mask_generator = RandomMaskingGenerator(
        input_size=args.global_crops_size // args.patch_size,
    )

    collate_fn = partial(
        collate_data_and_cast_aug,
        mask_ratio=args.mask_ratio,
        mask_probability=args.mask_probability,
        dtype=torch.half,   # half precision
        n_tokens=n_tokens,
        mask_first_n=args.mask_first_n,
        mask_generator=mask_generator,
        clone_batch=args.clone_batch,
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_fn,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_fn,
    )

    mixup_fn = None

    print(f"Creating model: {args.model}")
    meta_arch_module = importlib.import_module(args.model)
    MetaArch = meta_arch_module.MetaArch

    model = MetaArch(args)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        if 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint

        missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, False)
        print('missing_keys: ', missing_keys)
        print('unexpected_keys: ', unexpected_keys)
        
    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')
            
    model.to(device)

    classifier = torch.nn.Linear(model.student.backbone.embed_dim * 5, args.nb_classes).to(device)

    classifier_without_ddp = classifier
    if args.distributed:
        classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.gpu], find_unused_parameters=True)
        classifier_without_ddp = classifier.module

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model.student.backbone,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    
    optimizer = create_optimizer(args, classifier_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = 0
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)

    model.requires_grad_(False)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_classifier_one_epoch(
            model, classifier, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args = args,
        )


        valid_stats = validate_classifier_one_epoch(model, classifier, data_loader_val, device, epoch)

        # torch.distributed.all_reduce(torch.ones(1), op=torch.distributed.ReduceOp.MEAN)

        # print('Completed')

        # sys.exit(0)

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                    'classifier': classifier_without_ddp.state_dict()
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'valid_{k}': v for k, v in valid_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

from torch.nn import functional as F   

@torch.inference_mode()
def validate_classifier_one_epoch(model: torch.nn.Module, classifier: torch.nn.Module, data_loader: Iterable, device: torch.device, epoch: int):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    print(f'{len(data_loader) = }')

    for data_iter_step, inputs_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for k, v in inputs_dict.items():
            if isinstance(v, torch.Tensor):
                # print(f'{k = }, {v.shape = }')
                inputs_dict[k] = v.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            # loss_dict = model(inputs_dict)
            global_crops = inputs_dict["collated_global_crops"]
            # cur_masks = masks if model.cfg.mask_probability > 0 else None
            with torch.inference_mode():
                features = model.student.backbone.get_intermediate_layers(
                    global_crops, 4, return_class_token=True
                )
                features = create_linear_input(features, 4)
            # print(features.shape)
            logits = classifier(features)
        labels = inputs_dict['collated_global_labels'].to(torch.int64) # shape: [batch_size, n_classes]
        # Compute accuracy
        pred = torch.argmax(logits, dim=1)  # 获取每个样本的预测类别索引
        target = labels.squeeze()  # 确保 labels 形状匹配
        
        correct = (pred == target).sum().item()  # 计算正确预测的样本数量
        total = labels.size(0)  # 样本总数
        accuracy = correct / total  # 计算准确率
        
        metric_logger.update(acc=accuracy)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_classifier_one_epoch(model: torch.nn.Module, classifier: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(0)
    print_freq = 10

    for data_iter_step, inputs_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # print(f"{inputs_dict['collated_global_labels'] = }")
        for k, v in inputs_dict.items():
            if isinstance(v, torch.Tensor):
                # print(f'{k = }, {v.shape = }')
                inputs_dict[k] = v.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            # with torch.inference_mode():
            #     with self.autocast_ctx():
            #         features = self.feature_model.get_intermediate_layers(
            #             images, self.n_last_blocks, return_class_token=True
            #         )
            # loss_dict = model(inputs_dict)
            global_crops = inputs_dict["collated_global_crops"]
            # cur_masks = masks if model.cfg.mask_probability > 0 else None
            # with torch.inference_mode():
            #     output = model.student.backbone(
            #         global_crops, masks=None, is_training=True
            #     )
            #     x_prenorm: torch.Tensor = output['x_prenorm']
            # x_mean = x_prenorm.mean(dim=1)
            # logits = classifier(x_mean)
            with torch.inference_mode():
                features = model.student.backbone.get_intermediate_layers(
                    global_crops, 4, return_class_token=True
                )
                features = create_linear_input(features, 4)
            # print(features.shape)
            logits = classifier(features)

        labels = inputs_dict['collated_global_labels'].to(torch.int64)
        
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model.student.backbone)

        metric_logger.update(loss=loss)
        # metric_logger.update(patch_loss=patch_loss_value)
        # metric_logger.update(token_loss=token_loss_value)
        # metric_logger.update(fea_loss=fea_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    loader_len = len(data_loader)

    for data_iter_step, inputs_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        for k, v in inputs_dict.items():
            if isinstance(v, torch.Tensor):
                inputs_dict[k] = v.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            loss_dict = model(inputs_dict)
        
        loss = loss_dict["loss"]
        patch_loss = loss_dict["patch_loss"]
        fea_loss = loss_dict["fea_loss"]
        token_loss = loss_dict["token_loss"]

        patch_loss_value = patch_loss.item()
        token_loss_value = token_loss.item()
        fea_loss_value = fea_loss.item()
        loss_value = loss.item()
        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model.module.student.backbone)

        metric_logger.update(loss=loss_value)
        metric_logger.update(patch_loss=patch_loss_value)
        metric_logger.update(token_loss=token_loss_value)
        metric_logger.update(fea_loss=fea_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
