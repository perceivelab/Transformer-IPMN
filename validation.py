# coding=utf-8
from __future__ import absolute_import, division, print_function

import numpy as np


import torch

from tqdm import tqdm

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, RocCurveDisplay
from utils.plot_conf_matrix import plot_confusion_matrix
from utils.averageMeter import AverageMeter
from utils.utils import compute_metrics, get_accuracy
import matplotlib
matplotlib.use('Agg')

def valid(args, logger, model, writer, phase, test_loader, global_step, num_classes, KEYS, inner_loop_idx):
    figure = None
    # Validation!
    eval_losses = AverageMeter()

    model.eval()
    
    logger.info(f"***** Running {phase} *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc=f"{phase}... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    
    for step, batch in enumerate(epoch_iterator):
        if args.dataset in ['MRI','MRI-BALANCED','MRI-BALANCED-3Classes', 'MRI-BALANCED-3Classes_Nested', 'MRI-EQUAL']:
            x = batch[KEYS[0]].to(args.device)
            if args.image_modality == 'LateFusion':
                x2 = batch[KEYS[1]].to(args.device)
                x = (x,x2)
            y = batch[KEYS[1]].squeeze().to(args.device).long()
        else:
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
        with torch.no_grad():
            logits, eval_loss = model(x, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
            all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)
        epoch_iterator.set_description(f"{phase}... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    
    accuracy_dict = get_accuracy(all_preds, all_label, accuracy_type = args.accuracy)
    
    conf_matrix = confusion_matrix(all_label, all_preds)
    class_names = np.arange(num_classes)
    figure = plot_confusion_matrix(conf_matrix, class_names=class_names)
    
    logger.info("\n")
    logger.info(f"{phase} Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info(f"{phase} Loss: %2.5f" % eval_losses.avg)
    
    if args.accuracy == 'both':
        if inner_loop_idx is not None:
            writer.add_scalar(f"{phase}/accuracy_simple/inner_loop{inner_loop_idx}", scalar_value=accuracy_dict['accuracy_simple'], global_step=global_step)
            writer.add_scalar(f"{phase}/accuracy_balanced/inner_loop{inner_loop_idx}", scalar_value=accuracy_dict['accuracy_balanced'], global_step=global_step)
            writer.add_figure(f'{phase}/conf_matrix/inner_loop{inner_loop_idx}', figure, global_step=global_step)
        else:
            writer.add_scalar(f"{phase}/accuracy_simple", scalar_value=accuracy_dict['accuracy_simple'], global_step=global_step)
            writer.add_scalar(f"{phase}/accuracy_balanced", scalar_value=accuracy_dict['accuracy_balanced'], global_step=global_step)
            writer.add_figure(f'{phase}/conf_matrix', figure, global_step=global_step)
    else:
        if args.accuracy == 'simple':
            k = 'accuracy_simple'
        else:
            k = 'accuracy_balanced'
        if inner_loop_idx is not None:
            writer.add_scalar(f"{phase}/accuracy_{args.accuracy}/inner_loop{inner_loop_idx}", scalar_value=accuracy_dict[k], global_step=global_step)
            writer.add_figure(f'{phase}/conf_matrix/inner_loop{inner_loop_idx}', figure, global_step=global_step)
        else:
            writer.add_scalar(f"{phase}/accuracy_{args.accuracy}", scalar_value=accuracy_dict[k], global_step=global_step)
            writer.add_figure(f'{phase}/conf_matrix', figure, global_step=global_step)
    
    if args.accuracy == 'both':
        return accuracy_dict, {'eval_loss': eval_losses.val}
    else:
        return accuracy_dict, {}
 