import math
import sys

import torch
import torch.nn.functional as F
import torchvision

import utils.misc as misc
import utils.lr_sched as lr_sched

import numpy as np

import wandb

def visualize( model, samples, mask ,pred ):

    samples = samples.detach().cpu()

    pred = model.unpatchify(pred)
    pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    samples = torch.einsum('nchw->nhwc', samples)

    # masked image
    im_masked = samples * (1 - mask)
    pred = pred * mask

    # MAE reconstruction pasted with visible patches
    im_paste = samples * (1 - mask) + pred * mask

    return samples, im_masked, pred, im_paste # original, masked, reconstruction, reconstruction + visible

def visualize2( sample, labels, pred ):
    np_samples = sample.to('cpu').detach().numpy().copy()
    np_labels = labels.to('cpu').detach().numpy().copy() * 255
    np_pred = ((torch.sigmoid(pred).to('cpu').detach().numpy().copy()) > 0.75) * 255

    np_labels = np.stack( [np.einsum('nchw->nhwc', np_labels)[:,:,:,1]] * 3, axis=3 )
    np_pred = np.stack( [np.einsum('nchw->nhwc', np_pred)[:,:,:,1]] * 3, axis=3 )
    np_labels = np.transpose( np_labels, [0,3,1,2] )
    np_pred = np.transpose( np_pred, [0,3,1,2] ).astype(np.float32)

    return  np_samples, np_labels, np_pred

def dice_loss( pred, target, smooth = 1 ):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim = 2).sum(dim = 2)
    loss = (1 - ((2 * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def calc_loss( pred, target, metrics=None, bce_weight=0.5):
    # Dice Loss と Categorical Cross Entropy を混ぜていい感じにする
    bce = F.binary_cross_entropy_with_logits( pred, target )
    pred_sig = torch.sigmoid( pred )
    dice = dice_loss(pred_sig, target)
    loss = bce *bce_weight + dice * (1 - bce_weight)

    loss_value = loss.item()
    if not math.isfinite(loss_value):
        print("bingo")
    
    return loss

def train_one_epoch(model: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # 原本为loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
            #loss, _, _ = model(samples)
            loss, pred, mask = model(samples)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        if args.wan_db :
            wandb.log(
                {
                    "loss": loss_value_reduce
                }
            )

    if log_writer is not None and (epoch + 1) % args.save_freq == 0 :
        samples, im_masked, pred, im_paste = visualize( model, samples, mask ,pred )
        
        image = []
        for idx in range(5):
            image += [
                        np.transpose(torch.as_tensor(samples[idx]), [2,0,1]), 
                        np.transpose(torch.as_tensor(im_masked[idx]), [2,0,1]), 
                        np.transpose(torch.as_tensor(pred[idx]), [2,0,1]),
                        np.transpose(torch.as_tensor(im_paste[idx]), [2,0,1])
                    ]
        img_grid = torchvision.utils.make_grid(image, nrow=4)
        log_writer.add_image(f'images', img_grid, epoch)

    if args.wan_db and (epoch + 1) % args.save_freq == 0 :
        samples, im_masked, pred, im_paste = visualize( model, samples, mask ,pred )            

        # 画像をTableに格納してログする場合のイメージ
        table = wandb.Table(columns=["idx", "original", "masked", "reconstruction", "reconstruction + visible"])

        for idx in range(5):
            table.add_data(
                0,
                wandb.Image(torchvision.transforms.functional.to_pil_image(torch.permute(samples[idx], (2,0,1)))),
                wandb.Image(torchvision.transforms.functional.to_pil_image(torch.permute(im_masked[idx], (2,0,1)))),
                wandb.Image(torchvision.transforms.functional.to_pil_image(torch.permute(pred[idx], (2,0,1)))),
                wandb.Image(torchvision.transforms.functional.to_pil_image(torch.permute(im_paste[idx], (2,0,1))))
            ) 
          
        wandb.log({f'epoch[{epoch}]': table})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_unet(model: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, lables) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        lables = lables.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            pred = model(samples)

        loss_multilabel = calc_loss(pred, lables, bce_weight=0.5)
        loss_value = loss_multilabel.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_multilabel /= accum_iter
        loss_scaler(loss_multilabel, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        if args.wan_db :
            wandb.log(
                {
                    "loss": loss_value_reduce
                }
            )

    if log_writer is not None and (epoch + 1) % args.save_freq == 0 :
        np_samples, np_labels, np_pred = visualize2( samples, lables, pred)
        image = []
        for idx in range(5):
            image += [
                        torch.as_tensor(np_samples[idx]), 
                        torch.as_tensor(np_labels[idx]), 
                        torch.as_tensor(np_pred[idx])
                    ]
        img_grid = torchvision.utils.make_grid(image, nrow=3)
        log_writer.add_image(f'train_images', img_grid, epoch)

    if args.wan_db and (epoch + 1) % args.save_freq == 0 :
        np_samples, np_labels, np_pred = visualize2( samples, lables, pred)

        # 画像をTableに格納してログする場合のイメージ
        table = wandb.Table(columns=["idx", "image", "label", "prediction"])

        for idx in range(5):
            table.add_data(
                0,
                wandb.Image(torchvision.transforms.functional.to_pil_image(torch.as_tensor(np_samples[idx]))),
                wandb.Image(torchvision.transforms.functional.to_pil_image(torch.as_tensor(np_labels[idx]))),
                wandb.Image(torchvision.transforms.functional.to_pil_image(torch.as_tensor(np_pred[idx])))
            ) 
         
        wandb.log({f'epoch[{epoch}]': table})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def eval_one_epoch_unet(model: torch.nn.Module,
                    data_loader,
                    device: torch.device, 
                    epoch: int,
                    log_writer=None,
                    args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch Val: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, lables) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)
        lables = lables.to(device, non_blocking=True)

        with torch.no_grad():
            pred = model(samples)

        loss_multilabel = calc_loss(pred, lables, bce_weight=0.5)
        loss_value = loss_multilabel.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_multilabel /= accum_iter
        
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('val_loss', loss_value_reduce, epoch_1000x)

        if args.wan_db :
            wandb.log(
                {
                    "loss": loss_value_reduce
                }
            )

    if log_writer is not None :
        np_samples, np_labels, np_pred = visualize2( samples, lables, pred)
        image = []
        for idx in range(5):
            image += [
                        torch.as_tensor(np_samples[idx]), 
                        torch.as_tensor(np_labels[idx]), 
                        torch.as_tensor(np_pred[idx])
                    ]
        img_grid = torchvision.utils.make_grid(image, nrow=3)
        log_writer.add_image(f'eval_images', img_grid, epoch)

        if args.wan_db :
            # 画像をTableに格納してログする場合のイメージ
            table = wandb.Table(columns=["idx", "original", "label", "prediction"])

            for idx in range(5):
                table.add_data(
                    0,
                    wandb.Image(torchvision.transforms.functional.to_pil_image(torch.as_tensor(np_samples[idx]))),
                    wandb.Image(torchvision.transforms.functional.to_pil_image(torch.as_tensor(np_labels[idx]))),
                    wandb.Image(torchvision.transforms.functional.to_pil_image(torch.as_tensor(np_pred[idx])))
                ) 
            
            wandb.log({f'epoch[{epoch}]': table})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
