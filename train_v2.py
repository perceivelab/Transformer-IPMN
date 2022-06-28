# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import json

from datetime import timedelta, datetime

import torch
import time
#import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
#from apex import amp
#from apex.parallel import DistributedDataParallel as DDP

from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from models.modeling import VisionTransformer, LateFusionVisionTransformer, CONFIGS
#from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader, get_loss_weights
from utils.plot_conf_matrix import plot_confusion_matrix
#from utils.dist_util import get_world_size
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_accuracy(preds, labels, accuracy_type='simple'):
    if accuracy_type == 'simple':
        return (preds == labels).mean()
    elif accuracy_type == 'balanced':
        return balanced_accuracy_score(labels, preds)
    elif accuracy_type == 'both':
        simple_accuracy = (preds == labels).mean()
        balanced_accuracy = balanced_accuracy_score(labels, preds)
        return simple_accuracy, balanced_accuracy

def save_model(args, model, suffix='', global_step = None, sim_acc = None, bal_acc = None, test_acc_sim = None, test_acc_bal = None):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, f"{args.name}_{suffix}.bin")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info(f"Saved model {suffix} to [DIR: {args.output_dir}]")
    
    info = {
        'global_step': global_step,
        'simple_accuracy':sim_acc,
        'balanced_accuracy': bal_acc,
        'test_simple_accuracy': test_acc_sim,
        'test_balanced_accuracy': test_acc_bal,
        }
    
    with open(os.path.join(args.output_dir, f'{suffix}.json'), 'w') as fp:
        json.dump(info, fp)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    if args.image_modality in ['T1', 'T2', 'LateFusion']:
        in_channels = 1
    elif args.image_modality in ['EarlyFusion']:
        in_channels = 2
    else :
        in_channels = 3
        
    if args.dataset in ['MRI', 'MRI-BALANCED', 'MRI-EQUAL']:
        num_classes = 4
    elif args.dataset in ['MRI-BALANCED-3Classes', 'MRI-BALANCED-3Classes_2Val', 'MRI-BALANCED-3Classes_Nested']:
        num_classes = 3
    elif args.dataset == 'cifar10':
        num_classes = 10
    else: 
        num_classes = 100
    
    if args.image_modality == 'LateFusion':
        model = LateFusionVisionTransformer(config, args.img_size, in_channels=in_channels, zero_head=True, num_classes=num_classes, loss_weights = args.loss_weights, multi_stage_classification= args.multi_stage_classification, multi_layer_classification=args.multi_layer_classification)
    else:
        model = VisionTransformer(config, args.img_size, in_channels=in_channels, zero_head=True, num_classes=num_classes, loss_weights = args.loss_weights, multi_stage_classification= args.multi_stage_classification, multi_layer_classification=args.multi_layer_classification)
    if args.pretrained:
        model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return config, args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, phase, test_loader, global_step, KEYS, inner_loop_idx):
    # Validation!
    eval_losses = AverageMeter()

    logger.info(f"***** Running {phase} *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc=f"{phase}... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss(weight = args.loss_weights.to(args.device))
    for step, batch in enumerate(epoch_iterator):
        if args.dataset in ['MRI','MRI-BALANCED','MRI-BALANCED-3Classes', 'MRI-BALANCED-3Classes_2Val','MRI-BALANCED-3Classes_Nested', 'MRI-EQUAL']:
            x = batch[KEYS[0]].to(args.device)
            if args.image_modality == 'LateFusion':
                x2 = batch[KEYS[1]].to(args.device)
                x = (x,x2)
            y = batch[KEYS[-1]].squeeze().to(args.device).long()
        else:
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
        with torch.no_grad():
            if not args.multi_stage_classification:
                logits = model(x)[0]
                eval_loss = loss_fct(logits, y)
                eval_losses.update(eval_loss.item())
    
                preds = torch.argmax(logits, dim=-1)
            else:
                logits, _, logits2, index = model(x)
                preds = torch.argmax(logits, dim=-1)
                preds[preds == 1] = 2
                if logits2 is not None:
                    idxs = torch.tensor(preds == 0)
                    assert torch.all(index.eq(idxs)), 'Index of elements classified as 0 mismatch.'
                    preds2 = torch.argmax(logits2, dim=-1)
                    preds[index.tolist()] = preds2

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        if not args.multi_stage_classification:
            epoch_iterator.set_description(f"{phase}... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    if args.accuracy == 'both':
        simple_accuracy, balanced_accuracy = get_accuracy(all_preds, all_label, accuracy_type = args.accuracy)
    else:
        accuracy = get_accuracy(all_preds, all_label, accuracy_type = args.accuracy)
    
    conf_matrix = confusion_matrix(all_label, all_preds)
    class_names = np.arange(model.num_classes)
    figure = plot_confusion_matrix(conf_matrix, class_names=class_names)

    logger.info("\n")
    logger.info(f"{phase} Results")
    logger.info("Global Steps: %d" % global_step)
    if not args.multi_stage_classification:
        logger.info(f"{phase} Loss: %2.5f" % eval_losses.avg)
    if args.accuracy == 'both':
        logger.info(f"{phase} Simple Accuracy: %2.5f" % simple_accuracy)
        logger.info(f"{phase} Balanced_Accuracy: %2.5f" % balanced_accuracy)
    else:
        logger.info(f"{phase} Accuracy: %2.5f" % accuracy)
    
    if args.accuracy == 'both':
        if inner_loop_idx is not None:
            writer.add_scalar(f"{phase}/accuracy_simple/inner_loop{inner_loop_idx}", scalar_value=simple_accuracy, global_step=global_step)
            writer.add_scalar(f"{phase}/accuracy_balanced/inner_loop{inner_loop_idx}", scalar_value=balanced_accuracy, global_step=global_step)
            writer.add_figure(f'{phase}/conf_matrix/inner_loop{inner_loop_idx}', figure, global_step=global_step)
        else:
            writer.add_scalar(f"{phase}/accuracy_simple", scalar_value=simple_accuracy, global_step=global_step)
            writer.add_scalar(f"{phase}/accuracy_balanced", scalar_value=balanced_accuracy, global_step=global_step)
            writer.add_figure(f'{phase}/conf_matrix', figure, global_step=global_step)
    else:
        if inner_loop_idx is not None:
            writer.add_scalar(f"{phase}/accuracy_{args.accuracy}/inner_loop{inner_loop_idx}", scalar_value=accuracy, global_step=global_step)
            writer.add_figure(f'{phase}/conf_matrix/inner_loop{inner_loop_idx}', figure, global_step=global_step)
        else:
            writer.add_scalar(f"{phase}/accuracy_{args.accuracy}", scalar_value=accuracy, global_step=global_step)
            writer.add_figure(f'{phase}/conf_matrix', figure, global_step=global_step)
    
    if args.accuracy == 'both':
        return simple_accuracy, balanced_accuracy
    else:
        return accuracy

def train(args, model, KEYS, info = '', inner_loop_idx = None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        folders_logs = args.output_dir.split(os.path.sep)[1:]
        sub_path_logs = ''
        for s in folders_logs:
            sub_path_logs = os.path.join(sub_path_logs, s)
        writer = SummaryWriter(log_dir=os.path.join("logs", sub_path_logs))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    
    writer.add_text('info', str(info))

    # Prepare dataset
    train_loader, val_loader, test_loader = get_loader(args, inner_loop_idx)
    
    # Prepare optimizer and scheduler
    if args.optimizer == 'SGD':
        if args.train_only_classifier:
            optimizer = torch.optim.SGD(model.head.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        if args.train_only_classifier:
            optimizer = torch.optim.Adam(model.head.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        if args.train_only_classifier:
            optimizer = torch.optim.RMSprop(model.head.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.RMSprop(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
        
    t_total = args.num_steps
    '''
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    '''
    if args.fp16: #at the moment, amp is disabled
        '''
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20
        '''
    # Distributed training
    if args.local_rank != -1:
        '''
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
        '''

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    losses_1 = AverageMeter()
    losses_2 = AverageMeter()
    global_step, best_acc, best_simple_acc = 0, 0, 0
    while True:
        model.train()
        if args.train_only_classifier:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
        else:
           for param in model.parameters():
                param.requires_grad = True 
        if inner_loop_idx is not None:
            epoch_iterator = tqdm(train_loader,
                                  desc="InnerLoop X - Training (X / X Steps) (loss=X.X)",
                                  bar_format="{l_bar}{r_bar}",
                                  dynamic_ncols=True,
                                  disable=args.local_rank not in [-1, 0])
        else:
            epoch_iterator = tqdm(train_loader,
                                  desc="Training (X / X Steps) (loss=X.X)",
                                  bar_format="{l_bar}{r_bar}",
                                  dynamic_ncols=True,
                                  disable=args.local_rank not in [-1, 0])
        sum_train_accuracy = 0
        for step, batch in enumerate(epoch_iterator):
            if args.dataset in ['MRI','MRI-BALANCED','MRI-BALANCED-3Classes', 'MRI-BALANCED-3Classes_2Val', 'MRI-BALANCED-3Classes_Nested','MRI-EQUAL']:
                x = batch[KEYS[0]].to(args.device)
                if args.image_modality == 'LateFusion':
                    x2 = batch[KEYS[1]].to(args.device)
                    x = (x,x2)
                y = batch[KEYS[-1]].to(args.device).long()
            else:
                batch = tuple(t.to(args.device) for t in batch)
                x, y = batch
            if not args.multi_stage_classification:
                logits, loss = model(x, y)
                pred_labels = logits.argmax(-1)
            else:
                logits, logits2, loss1, loss2, index = model(x, y)
                loss = loss1 + loss2
                
                pred_labels = torch.argmax(logits, dim=-1)
                idxs = torch.flatten(y) != 2
                assert torch.all(index.eq(idxs)), 'Index of elements mismatch.'
                preds2 = torch.argmax(logits2, dim=-1)
                pred_labels[pred_labels == 1] = 2
                pred_labels[index.tolist()] = preds2
            #compute accuracy
            
            if args.accuracy == 'both':
                accuracy_type = 'balanced'
            else:
                accuracy_type= args.accuracy
            batch_accuracy = get_accuracy(pred_labels.detach().cpu().numpy(), y.detach().cpu().numpy(), accuracy_type = accuracy_type)
            sum_train_accuracy += batch_accuracy

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                if args.multi_stage_classification:
                    loss1 = loss1 / args.gradient_accumulation_steps
                    loss2 = loss2 / args.gradient_accumulation_steps
            if args.fp16:
                '''
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                '''
            else:
                loss.backward()
                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.multi_stage_classification:
                    losses_1.update(loss1.item()*args.gradient_accumulation_steps)
                    losses_2.update(loss2.item()*args.gradient_accumulation_steps)
                '''
                if args.fp16:
                      
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                '''
                
                '''
                # Add Tensorboard histograms of last classification layers params
                for name, param in model.head.named_parameters():
                    writer.add_histogram(f"{name}_grad", param.grad.data, global_step)
                '''
                optimizer.step()
                #scheduler.step()
                
                optimizer.zero_grad()
                global_step += 1
                
                if inner_loop_idx is not None:
                    epoch_iterator.set_description(
                        "InnerLoop %d - Training (%d / %d Steps) (loss=%2.5f)" % (inner_loop_idx, global_step, t_total, losses.val)
                    )
                else:
                    epoch_iterator.set_description(
                        "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                    )
                if args.local_rank in [-1, 0]:
                    if inner_loop_idx is not None:
                        writer.add_scalar(f"train/loss/inner_loop{inner_loop_idx}", scalar_value=losses.val, global_step=global_step)
                    else:
                        writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    if args.multi_stage_classification:
                        if inner_loop_idx is not None:
                            writer.add_scalar(f"train_multi_stage/loss1/inner_loop{inner_loop_idx}", scalar_value=losses_1.val, global_step=global_step)
                            writer.add_scalar(f"train_multi_stage/loss2/inner_loop{inner_loop_idx}", scalar_value=losses_2.val, global_step=global_step)
                        else:
                            writer.add_scalar("train_multi_stage/loss1", scalar_value=losses_1.val, global_step=global_step)
                            writer.add_scalar("train_multi_stage/loss2", scalar_value=losses_2.val, global_step=global_step)
                    #writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    '''
                    # Add Tensorboard histograms of last classification layers params
                    for name, param in model.head.named_parameters():
                        writer.add_histogram(name, param.data, global_step)
                    '''
                    
                    #validation
                    val_accuracy = valid(args, model, writer, "validation", val_loader, global_step, KEYS, inner_loop_idx)
                    sim_acc, bal_acc = val_accuracy
                    #test
                    test_accuracy = valid(args, model, writer, "test", test_loader, global_step, KEYS, inner_loop_idx)
                    if args.accuracy == 'both':
                         #balanced
                        save_model(args, model, 'checkpoint', global_step, sim_acc, bal_acc, test_accuracy[0], test_accuracy[1])
                        if best_acc <= bal_acc:
                            save_model(args, model, 'best', global_step, sim_acc, bal_acc, test_accuracy[0], test_accuracy[1])
                            best_acc = bal_acc
                        if best_simple_acc <= sim_acc:
                            save_model(args, model, 'best_simple', global_step, sim_acc, bal_acc, test_accuracy[0], test_accuracy[1])
                            best_simple_acc = sim_acc
                    elif args.accuracy == 'simple':           
                        save_model(args, model, 'checkpoint', global_step, sim_acc, bal_acc, test_accuracy[0], test_accuracy[1])
                        if best_simple_acc <= sim_acc:
                            save_model(args, model, 'best_simple', global_step, sim_acc, bal_acc, test_accuracy[0], test_accuracy[1])
                            best_simple_acc = sim_acc
                    else:
                        save_model(args, model, 'checkpoint', global_step, sim_acc, bal_acc, test_accuracy[0], test_accuracy[1])
                        if best_acc <= bal_acc:
                            save_model(args, model, 'best', global_step, sim_acc, bal_acc, test_accuracy[0], test_accuracy[1])
                            best_acc = bal_acc
                    
                    model.train()

                if global_step % t_total == 0:
                    break
         
        epoch_train_accuracy = sum_train_accuracy / len(train_loader)
        if inner_loop_idx is not None:
            writer.add_scalar(f"train/accuracy_{accuracy_type}", scalar_value = epoch_train_accuracy, global_step=global_step)
        else:
            writer.add_scalar(f"train/accuracy_{accuracy_type}/inner_loop{inner_loop_idx}", scalar_value = epoch_train_accuracy, global_step=global_step)
            
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", 'MRI', 'MRI-BALANCED', 'MRI-BALANCED-3Classes', 'MRI-BALANCED-3Classes_2Val', 'MRI-EQUAL', 'MRI-BALANCED-3Classes_Nested'], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--image_modality", choices=['T1', 'T2', 'EarlyFusion', 'EarlyFusion3Channels', 'LateFusion', 'T1_nopatched', 'T2_nopatched'],
                        help="Which image modality use for MRI.")
    parser.add_argument("--accuracy", choices=['simple','balanced', 'both'], default='both', 
                        help="The type of accuracy computed")
    parser.add_argument("--num_fold", choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], type=int, required=True,
                        help="Which Cross Validation folder use.")
    parser.add_argument("--num_inner_loop", type=int, default= 9,
                        help="Number of fold of inner loop, for Nested train.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16", 
                                                 "ViT-MRI", 'R50-ViT-MRI'],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained", action='store_true' ,
                        help="If use a pretrained model.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--train_only_classifier", action='store_true',
                        help="Freeze all network and train only the classifier")
    parser.add_argument("--multi_stage_classification", action='store_true',
                        help="if performing a multi-stage classification")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=192, type=int,
                        help="Resolution size")
    parser.add_argument("--num_patches", default=9, type=int,
                        help="Number of patches to create a patched input")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--test_batch_size", default=16, type=int,
                        help="Total batch size for test.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    
    parser.add_argument("--optimizer", choices=["SGD", "Adam", 'RMSprop'], default='SGD', type=str,
                        help="The optimizer.")
    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    '''
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    '''
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--cuda_id", type=int, default=0,
                        help="Index of GPU")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    
    parser.add_argument('--multi_layer_classification', action='store_true',
                        help = "whether a multi layer classification is used or not.")
    
    
    args = parser.parse_args()
    
    
    args.loss_weights = get_loss_weights(args.dataset, args.multi_stage_classification)
    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    
    timestamp_str = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    args.name = timestamp_str + args.name + '_fold' + str(args.num_fold)
    
    args.output_dir = os.path.join(args.output_dir, args.name)
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    if args.image_modality in  ['T1', 'T1_nopatched']:
        KEYS = ('image_T1', 'label')
    elif args.image_modality in  ['T2', 'T2_nopatched']:
        KEYS = ('image_T2', 'label')
    elif args.image_modality in ['EarlyFusion', 'EarlyFusion3Channels']:
        KEYS = ('fusedImage', 'label')
    elif args.image_modality in ['LateFusion']:
        KEYS = ('image_T1', 'image_T2', 'label')
    
    dict_args = vars(args).copy()
    dict_args['device'] = str(dict_args['device'])
    dict_args['loss_weights'] = dict_args['loss_weights'].tolist()
    # Saving training info
    
    if args.dataset != 'MRI-BALANCED-3Classes_Nested':
        # Model & Tokenizer Setup
        config, args, model = setup(args)
        
        info = {
            'model_name': model.__class__.__name__,
            'KEYS': KEYS,
            'model_config': config.to_dict(), 
            'model_args': dict_args
            }
        
        with open(os.path.join(args.output_dir, args.name+'.json'), 'w') as fp:
            json.dump(info, fp)
    
        # Training
        train(args, model, KEYS, info = info)
    else:
        out_dir = args.output_dir
        for i in range(args.num_inner_loop):
            args.output_dir = os.path.join(out_dir, f'inner_loop_{i}')
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir)
            config, args, model = setup(args)
            info = {
                'model_name': model.__class__.__name__,
                'KEYS': KEYS,
                'model_config': config.to_dict(), 
                'model_args': dict_args
                }
            with open(os.path.join(args.output_dir, args.name+'.json'), 'w') as fp:
                json.dump(info, fp)
    
            # Training
            train(args, model, KEYS, inner_loop_idx = i)
            

if __name__ == "__main__":
    main()
