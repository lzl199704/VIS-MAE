import math
import sys

import torch

import utils.misc as misc
import utils.lr_sched as lr_sched
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(1, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
        
        
def dice_score(pred, target, epsilon=1e-6):
    # Apply softmax to the output to get class probabilities
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    # Convert to one-hot format
    pred = torch.argmax(pred, dim=1)
    #print(num_classes)
    
    if target.dim() == 4 and target.size(1) == 1:  
        target = target.squeeze(1)
    target=target.long()
    pred_one_hot = F.one_hot(pred, num_classes=num_classes).permute(0, 3, 1, 2)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)

    
    target_one_hot = target_one_hot[:, 1:]  # Exclude the first channel
    pred_one_hot = pred_one_hot[:, 1:]  # Exclude the first channel
    # Compute Dice Score per class
    dice_scores = []

    # Calculate Dice score per class
    for class_idx in range(num_classes-1):
        intersect = (pred_one_hot[:, class_idx, :, :] * target_one_hot[:, class_idx, :, :]).sum()
        union = pred_one_hot[:, class_idx, :, :].sum() + target_one_hot[:, class_idx, :, :].sum()

        # Compute class-specific Dice score
        dice_score_class = (2. * intersect + epsilon) / (union + epsilon)
        dice_scores.append(dice_score_class)

    # Exclude certain classes if needed, e.g., background class
    # dice_scores = [score for i, score in enumerate(dice_scores) if i not in excluded_classes]

    # Calculate the average Dice score across all classes
    avg_dice_score = torch.stack(dice_scores).mean()

    return avg_dice_score
    
    
    #y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=10)[...,1:])
    #y_pred_f = K.flatten(y_pred[...,1:])
    #intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    #denom = K.sum(y_true_f + y_pred_f, axis=-1)
    #return K.mean((2. * intersect / (denom + smooth)))

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, dice_loss: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('train_dice', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.logdir))

    for data_iter_step, (samples, targets,_) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
            #loss, _, _ = model(samples)
            outputs = model(samples)
            #print(outputs.shape,target.shape)
            targets = targets.squeeze(1)
            #print(outputs.shape,targets.shape)
            loss_ce = criterion(outputs, targets)

        dice_value = dice_score(outputs, targets)
        loss_dice = dice_loss(outputs, targets, softmax=True)
        loss = 0.4 * loss_ce + 0.6 * loss_dice
        
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
        metric_logger.update(train_dice=dice_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        #if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        #    """ We use epoch_1000x as the x-axis in tensorboard.
        #    This calibrates different curves when batch size changes.
        #    """
       #     epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
       #     log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
       #     log_writer.add_scalar('lr', lr, epoch_1000x)
       #     log_writer.add_scalar('train_dice', dice_value, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, save_img, output_dir,dice_loss):
    criterion = torch.nn.CrossEntropyLoss()
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    if save_img==False:
        header = 'Validation:'
    else:
        header = 'Test'

    # switch to evaluation mode
    model.eval()
    if save_img==True:
            if os.path.exists(output_dir+'/prediction')==False:
                os.makedirs(output_dir+'/prediction')
            else:
                os.system('rm -r '+output_dir+'/prediction/*')
            if os.path.exists(output_dir+'/ground_truth')==False:
                os.makedirs(output_dir+'/ground_truth')
            else:
                os.system('rm -r '+output_dir+'/ground_truth/*')
            if os.path.exists(output_dir+'/images')==False:
                os.makedirs(output_dir+'/images')
            else:
                os.system('rm -r '+output_dir+'/images/*')
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        img_names = batch[2]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss_ce = criterion(output, target)
        
        # Calculate Dice score
            loss_dice = dice_loss(output, target, softmax=True)
        loss = 0.4 * loss_ce + 0.6 * loss_dice
        
        dice_score_val = dice_score(output, target)
        #print(dice_score_val)
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['dice_score'].update(dice_score_val.item(), n=batch_size)
        
        if save_img==True:
            
            #print(output.shape,output[0,0,:,:].max(),output[0,1,:,:].max(),output[0,2,:,:].max())
            pred_masks = torch.argmax(F.softmax(output, dim=1), dim=1)
            
            for pred_mask, img_name in zip(pred_masks, img_names):
                # Convert to PIL image and save
                pred_mask = pred_mask.cpu().numpy()
                #print(np.unique(pred_mask), torch.unique(target))
                pred_mask_img = Image.fromarray(pred_mask.astype(np.uint8))
                
                target_img = target.cpu().squeeze().numpy()
                #print(target_img.shape)
                target_img = Image.fromarray(target_img.astype(np.uint8))
                
                images_img = images.cpu().squeeze().numpy()
                images_img = images_img.transpose(1, 2, 0)
                #print(images_img.shape)
                images_img = Image.fromarray(images_img.astype(np.uint8))

                
                base_name = img_name.split('/')[-1]
                save_path = os.path.join(output_dir+'/prediction', base_name)
                pred_mask_img.save(save_path)
                
                save_path1 = os.path.join(output_dir+'/ground_truth', base_name)
                target_img.save(save_path1)
                
                save_path2 = os.path.join(output_dir+'/images', base_name)
                images_img.save(save_path2)
        
    metric_logger.synchronize_between_processes()
    print('* Dice Score {dice.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(dice=metric_logger.dice_score, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}