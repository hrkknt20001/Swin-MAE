import math
import sys

import torch
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

    if log_writer is not None :
        samples, im_masked, pred, im_paste = visualize( model, samples, mask ,pred )
        
        image = []
        for idx in range(5):
            image += [
                        np.transpose(torch.as_tensor(samples[idx]), [2,0,1]), 
                        np.transpose(torch.as_tensor(im_masked[idx]), [2,0,1]), 
                        np.transpose(torch.as_tensor(pred[idx]), [2,0,1]),
                        np.transpose(torch.as_tensor(im_paste[idx]), [2,0,1])
                    ]
        img_grid = torchvision.utils.make_grid(image, nrow=5)
        log_writer.add_image(f'images', img_grid, epoch)

        if args.wan_db :
            #sample, im_masked, pred, im_paste = visualize( model, samples, mask ,pred )

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
            
            wandb.log({f'epoch[{epoch}]]': table})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
