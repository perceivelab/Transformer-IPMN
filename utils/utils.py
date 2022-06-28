import os
import torch
import json
import random
import numpy as np

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

def compute_metrics(labels, preds, multi_label = False):
    if not multi_label:
        metrics = {
            'precision' : precision_score(labels, preds, average='macro', zero_division = 0),
            'recall' : recall_score(labels, preds, average='macro', zero_division = 0),
            'f1_score' : f1_score(labels, preds, average='macro', zero_division = 0),
            'jaccard_score': jaccard_score(labels, preds, average='macro', zero_division = 0)
            }
    else:
        metrics = {
            'precision_macro' : precision_score(labels, preds, average='macro', zero_division = 0),
            'precision_micro' : precision_score(labels, preds, average='micro', zero_division = 0),
            'recall_macro' : recall_score(labels, preds, average='macro', zero_division = 0),
            'recall_micro' : recall_score(labels, preds, average='micro', zero_division = 0),
            'f1_score_macro' : f1_score(labels, preds, average='macro', zero_division = 0),
            'f1_score_micro' : f1_score(labels, preds, average='micro', zero_division = 0),
            'jaccard_score_macro': jaccard_score(labels, preds, average='macro', zero_division = 0),
            'jaccard_score_micro': jaccard_score(labels, preds, average='micro', zero_division = 0)
            }
    return metrics

def get_accuracy(preds, labels, accuracy_type='simple'):
    if accuracy_type == 'simple':
        {'accuracy_simple': (preds == labels).mean()}
    elif accuracy_type == 'balanced':
        return {'accuracy_balanced':balanced_accuracy_score(labels, preds)}
    elif accuracy_type == 'both':
        simple_accuracy = (preds == labels).mean()
        balanced_accuracy = balanced_accuracy_score(labels, preds)
        return {'accuracy_simple':simple_accuracy, 'accuracy_balanced':balanced_accuracy}

def save_model(args, logger, model, suffix='', global_step = None, val_accuracy_dict = None, test_accuracy_dict = None, val_roc_metrics_dict = None, test_roc_metrics_dict = None):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, f"{args.name}_{suffix}.bin")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info(f"Saved model {suffix} to [DIR: {args.output_dir}]")
    
    info = {
        'global_step': global_step
        }
    
    if not val_accuracy_dict is None:
        for k in val_accuracy_dict.keys():
            info[f'val_{k}'] = val_accuracy_dict[k]
        
    if not test_accuracy_dict is None:
        for k in test_accuracy_dict.keys():
            info[f'test_{k}'] = test_accuracy_dict[k]
            
    if not val_roc_metrics_dict is None:
        for k in val_roc_metrics_dict.keys():
            info[f'val_roc_{k}'] = val_roc_metrics_dict[k]
            
    if not test_roc_metrics_dict is None:
        for k in test_roc_metrics_dict.keys():
            info[f'test_roc_{k}'] = test_roc_metrics_dict[k]
        
    with open(os.path.join(args.output_dir, f'{suffix}.json'), 'w') as fp:
        json.dump(info, fp)
        
    with open(os.path.join(args.output_dir, f'log_{suffix}.json'), 'a') as f:
        json.dump(info, f, indent = 0)
        

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

