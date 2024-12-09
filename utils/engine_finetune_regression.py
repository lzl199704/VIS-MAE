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
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import label_binarize
import pandas as pd

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

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, 
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
            
            #print(outputs.shape,target.shape)
            loss = criterion(outputs, targets)


        
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

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()

def evaluate(data_loader, model, device, output_dir):
    criterion = torch.nn.L1Loss()  # Using MSE loss for regression

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test' if output_dir != '' else 'Validation:'

    # switch to evaluation mode
    model.eval()
    results = []
    all_preds = []
    all_targets = []
    all_image_names = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        targets = batch[1]
        image_names = batch[2]
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Collect predictions, targets, and image names
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_image_names.extend(image_names)

        metric_logger.update(loss=loss.item())

    # Calculate MSE for the entire dataset
    mae_score = mean_absolute_error(all_targets, all_preds)
    print(f"MAE Score = {mae_score:.4f}")

    # Prepare results for DataFrame
    results = [{
        'image_name': img_name,
        'ground_truth': gt,
        'prediction': pred[0]  # Assuming output is single value per item
    } for img_name, gt, pred in zip(all_image_names, all_targets, all_preds)]

    metric_logger.synchronize_between_processes()
    print('Test MAE Score:', mae_score)

    if output_dir != '':
        df_results = pd.DataFrame(results)
        df_results.to_csv(f"{output_dir}/results_pred.csv", index=False)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    
    