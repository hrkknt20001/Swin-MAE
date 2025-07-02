import argparse
import sys
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from PIL import Image

from .. import swin_mae

# MAEのエンコーダ部分を抽出して、セグメンテーションに利用
class SwinMAE_Segmentation(nn.Module):
    def __init__(self, mae_encoder, num_classes=21):  # 21クラスを例とする（Pascal VOCなど）
        super(SwinMAE_Segmentation, self).__init__()
        self.encoder = mae_encoder
        self.segmentation_head = nn.Conv2d(768, num_classes, kernel_size=1)  # 768はMAEの出力次元
        
    def forward(self, x):
        # MAEのエンコーダ部分を通過させる
        encoded_feats = self.encoder(x)
        
        # セグメンテーションヘッドで出力マップを生成
        segmentation_map = self.segmentation_head(encoded_feats)
        
        return segmentation_map
    
def get_args_parser():
    parser = argparse.ArgumentParser('SwinMAE Segmentation-training', add_help=False)

    # common parameters
    parser.add_argument('--batch_size', default=96, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_freq', default=400, type=int)
    parser.add_argument('--checkpoint_encoder', default='', type=str)
    parser.add_argument('--checkpoint_decoder', default='', type=str)
    parser.add_argument('--data_path', default=r'C:\文件\数据集\腮腺对比学习数据集\三通道合并\concat\train', type=str)  # fill in the dataset path here
    parser.add_argument('--data_class', default=r'ImageFolder', type=str)
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # model parameters
    parser.add_argument('--model', default='swin_mae', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

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
    parser.add_argument('--wan_db', default=False)
    return parser


def prepare_model(chkpt_dir_, arch='swin_mae'):
    # build model
    model = getattr(swin_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir_, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

if __name__ == '__main__':
    # 读取图像
    img_root = r'D:\文件\数据集\腮腺对比学习数据集\三通道合并\concat\临时取出'
    img_name = r'135_6_l.png'
    img = Image.open(os.path.join(img_root, img_name))
    img = img.resize((224, 224))
    img = np.asarray(img) / 255.
    assert img.shape == (224, 224, 3)

    # 读取模型
    chkpt_dir = r'output_dir\checkpoint-400.pth'
    model_mae = prepare_model(chkpt_dir, 'swin_mae')
    print('Model loaded.')

    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    print('MAE with pixel reconstruction:')
    run_one_image(img, model_mae)
