# coding=utf-8
from __future__ import absolute_import, division, print_function

import os


import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from validation import valid
from utils.data_utils import get_loader
from utils.averageMeter import AverageMeter
from utils.utils import get_accuracy, save_model
import matplotlib
matplotlib.use('Agg')


def train(args, logger, model, KEYS, info = '', inner_loop_idx = None):
    
    folders_logs = args.output_dir.split(os.path.sep)[1:]
    sub_path_logs = ''
    for s in folders_logs:
        sub_path_logs = os.path.join(sub_path_logs, s)
    writer = SummaryWriter(log_dir=os.path.join("logs", sub_path_logs))
    
    writer.add_text('info', str(info))
    
    train_loader, val_loader, test_loader = get_loader(args)
    
    if args.train_only_classifier:
        params = model.head.parameters()
    else:
        params = model.parameters()
    
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params,
                                    lr=args.learning_rate,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params,
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(params,
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    model.zero_grad()
    
    t_total = args.num_steps
    losses = AverageMeter()
    global_step, best_acc, best_simple_acc = 0, 0, 0
    
    while True:
        model.train()
        if args.train_only_classifier:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
        else:
           for param in model.parameters():
                param.requires_grad = True
        if inner_loop_idx is not None:
            epoch_iterator = tqdm(train_loader,
                                  desc="InnerLoop X - Training (X / X Steps) (loss=X.X)",
                                  bar_format="{l_bar}{r_bar}",
                                  dynamic_ncols=True,
                                  )
        else:
            epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              )
        sum_train_accuracy = 0
        
        for step, batch in enumerate(epoch_iterator):
            if args.dataset in ['MRI','MRI-BALANCED','MRI-BALANCED-3Classes','MRI-BALANCED-3Classes_Nested', 'MRI-EQUAL']:
                x = batch[KEYS[0]].to(args.device)
                if args.image_modality == 'LateFusion':
                    x2 = batch[KEYS[1]].to(args.device)
                    x = (x,x2)
                y = batch[KEYS[-1]].to(args.device).long()
            else:
                batch = tuple(t.to(args.device) for t in batch)
                x, y = batch
                
            
            outputs, loss = model(x, y)
            
            #compute accuracy
            pred_labels = outputs.argmax(-1)
            
            if args.accuracy == "both":
                accuracy_type = "balanced" 
            else: accuracy_type = args.accuracy
            
            batch_accuracy = get_accuracy(pred_labels.detach().cpu().numpy(), y.squeeze().detach().cpu().numpy(), accuracy_type = accuracy_type)#(pred_labels == y.squeeze(dim = 1)).sum().item()/y.size(0)
            sum_train_accuracy += batch_accuracy['accuracy_'+accuracy_type]
            
            loss.backward()
            losses.update(loss.item())
            
            '''
            # Add Tensorboard histograms of last classification layers params
            for name, param in model.classifier.named_parameters():
                writer.add_histogram(f"{name}_grad", param.grad.data, global_step)
                writer.add_histogram(f"{name}_param", param.data, global_step)
            '''
            
            optimizer.step()
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

            if inner_loop_idx is not None:
                writer.add_scalar(f"train/loss/inner_loop{inner_loop_idx}", scalar_value=losses.val, global_step=global_step)
            else:
                writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
            
            
            if global_step % args.eval_every == 0:
                #validation
                val_accuracy_dict, val_loss_dict = valid(args, logger, model, writer, "validation", val_loader, global_step, args.num_classes, KEYS, inner_loop_idx)
                #test
                test_accuracy_dict, test_loss_dict = valid(args, logger, model, writer, "test", test_loader, global_step, args.num_classes, KEYS, inner_loop_idx)
                if args.accuracy == 'both':
                    bal_acc = val_accuracy_dict['accuracy_balanced']
                    sim_acc = val_accuracy_dict['accuracy_simple']
                     #balanced
                    save_model(args, logger, model, 'checkpoint', global_step, val_accuracy_dict = val_accuracy_dict, test_accuracy_dict= test_accuracy_dict)
                    if best_acc <= bal_acc:
                        save_model(args, logger, model, 'best', global_step, val_accuracy_dict = val_accuracy_dict, test_accuracy_dict= test_accuracy_dict)
                        best_acc = bal_acc
                    if best_simple_acc <= sim_acc:
                        save_model(args, logger, model, 'best_simple', global_step, val_accuracy_dict = val_accuracy_dict, test_accuracy_dict= test_accuracy_dict)
                        best_simple_acc = sim_acc
                elif args.accuracy == 'simple':           
                    save_model(args, logger, model, 'checkpoint', global_step, val_accuracy_dict = val_accuracy_dict, test_accuracy_dict= test_accuracy_dict)
                    sim_acc = val_accuracy_dict['accuracy_simple']
                    if best_simple_acc <= sim_acc:
                        save_model(args, logger, model, 'best_simple', global_step, sval_accuracy_dict = val_accuracy_dict, test_accuracy_dict= test_accuracy_dict)
                        best_simple_acc = sim_acc
                else:
                    bal_acc = val_accuracy_dict['accuracy_balanced']
                    save_model(args, logger, model, 'checkpoint', global_step, val_accuracy_dict = val_accuracy_dict, test_accuracy_dict= test_accuracy_dict)
                    if best_acc <= bal_acc:
                        save_model(args, logger, model, 'best', global_step, val_accuracy_dict = val_accuracy_dict, test_accuracy_dict= test_accuracy_dict)
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
        
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")