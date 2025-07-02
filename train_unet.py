import argparse
import json
import numpy as np
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
#import swin_mae
import swin_unet
from utils.engine_pretrain import train_one_epoch_unet, eval_one_epoch_unet

import GSI_Dataset
import wandb

from PIL import Image
import matplotlib.pyplot as plt

import albumentations as albu
import albumentations.pytorch

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # common parameters
    parser.add_argument('--batch_size', default=96, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_freq', default=400, type=int)
    parser.add_argument('--checkpoint_encoder', default='', type=str)
    parser.add_argument('--checkpoint_decoder', default='', type=str)
    parser.add_argument('--data_path_train', default=r'C:\文件\数据集\腮腺对比学习数据集\三通道合并\concat\train', type=str)  # fill in the dataset path here
    parser.add_argument('--data_path_eval', default=r'C:\文件\数据集\腮腺对比学习数据集\三通道合并\concat\train', type=str)  # fill in the dataset path here
    parser.add_argument('--target_class', default=r'17_Road', type=str)
    parser.add_argument('--data_class', default=r'ImageFolder', type=str)
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--data_augment', default=None, type=str)

    # model parameters
    parser.add_argument('--model', default='swin_unet', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    # optimizer parameters
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # other parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--wan_db', action='store_true')
    return parser


def main(args):
    # Fixed random seeds
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set up training equipment
    device = torch.device(args.device)
    cudnn.benchmark = True



    # Set dataset
    if args.data_class == 'GSI_Dataset':
        # Defining data augmentation
        transform_train = GSI_Dataset.get_augmentation(args.data_augment)
        dataset_train = GSI_Dataset.GSI_DatasetWithMask(args.data_path_train, args.target_class, transform=transform_train)
        transform_eval = GSI_Dataset.get_augmentation('Original')
        dataset_eval = GSI_Dataset.GSI_DatasetWithMask(args.data_path_eval, args.target_class, transform=transform_eval)
    else:
        # Defining data augmentation
        transform_train = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
        
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    sampler_eval = torch.utils.data.RandomSampler(dataset_eval)    
    data_loader_eval = torch.utils.data.DataLoader(
        dataset_eval, sampler=sampler_eval,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    # Log output
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Set model
    model = swin_unet.__dict__[args.model]()
    model.to(device)
    model_without_ddp = model

    #wandb.watch(model)

    # Set optimizer
    param_groups = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=arg.weight_decay, betas=(0.9, 0.95))  # 原来是5E-2
    loss_scaler = NativeScaler()

    # Create model
    misc.load_model(args=args, model_without_ddp=model_without_ddp)

    # Start the training process
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch_unet(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        eval_stats = eval_one_epoch_unet(
            model, data_loader_eval,
            device, epoch,         
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and ((epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch + 1)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.wan_db :
            wandb.log(
                {
                    "epoch_lr": log_stats["train_lr"],
                    "epoch_loss": log_stats["train_loss"]
                }
            )

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    arg = get_args_parser()
    arg = arg.parse_args()

    param = {
        "model" : arg.model,
        "epochs": arg.epochs,
        "batch_size": arg.batch_size,
        "learning_rate": arg.lr,
        "min_lr": arg.min_lr,
        "weight_decay": arg.weight_decay,
        "architecture": "Swin-MAE",
        "dataset": arg.data_class,
        "augment": arg.data_augment,
        "data_path_train": arg.data_path_train,
        "data_path_eval": arg.data_path_eval,
        "input_size": arg.input_size,
        "output_dir": arg.output_dir,
        "log_dir": arg.log_dir
    }

    # start a new wandb run to track this script
    if arg.wan_db :
        wandb.init(
            # set the wandb project where this run will be logged
            project="Swin-UNet eval project",

            dir = arg.log_dir,

            # track hyperparameters and run metadata
            config=param
        )

    if arg.output_dir:
        Path(arg.output_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(arg.output_dir, 'param.json'), 'w') as f:
        json.dump(param, f, indent=2)

    main(arg)

    if arg.wan_db :
        wandb.finish()
