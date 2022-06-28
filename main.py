from __future__ import absolute_import, division, print_function
import os
import torch


import json
import logging
import argparse
import numpy as np
from utils.data_utils import get_loss_weights
from datetime import datetime
import time
from utils.utils import count_parameters


from train import train

from models.modeling import CNN, CNNClassifier, VisionTransformer, LateFusionVisionTransformer, CONFIGS

logger = logging.getLogger(__name__)
    
def setup(args):
    if args.image_modality in ['T1', 'T2', 'LateFusion']:
        in_channels = 1
    elif args.image_modality in ['EarlyFusion']:
        in_channels = 2
    else :
        in_channels = 3

    if args.dataset in ['MRI', 'MRI-BALANCED', 'MRI-EQUAL']:
        num_classes = 4
    elif args.dataset in ['MRI-BALANCED-3Classes', 'MRI-BALANCED-3Classes_Nested']:
        num_classes = 3
    elif args.dataset == 'cifar10':
        num_classes = 10
    else: 
        num_classes = 100
        
    args.num_classes = num_classes
    args.in_channels = in_channels
    
    config = CONFIGS['None']
    if args.model == 'CNN':
        model = CNN(in_channels = in_channels, num_classes = num_classes).to(args.device)
    elif args.model in ['DenseNet', 'AlexNet', 'ResNet18','EfficientNet_b5', 'MobileNet_v2', 'CoAtNet_0', 'CoAtNet_1', 'CoAtNet_2', 'CoAtNet_3', 'CoAtNet_4']:
        model = CNNClassifier(in_channels = in_channels, num_classes = num_classes, loss_weights = args.loss_weights, model_type = args.model, pretrained = args.pretrained, img_size = args.img_size)
    elif args.model in ['VisionTransformer']:
        config = CONFIGS[args.model_type]
        if args.image_modality == 'LateFusion':
            model = LateFusionVisionTransformer(config, args.img_size, in_channels=in_channels, zero_head=True, num_classes=num_classes, loss_weights = args.loss_weights)
        else:
            model = VisionTransformer(config, args.img_size, in_channels=in_channels, zero_head=True, num_classes=num_classes, loss_weights = args.loss_weights)
        if args.pretrained:
            model.load_from(np.load(args.pretrained_dir))
    else:
        raise ValueError('Model type not supported!')
    
    model = model.to(args.device)

    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model, config

def main():
    
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", 'MRI-BALANCED', 'MRI-BALANCED-3Classes', 'MRI-BALANCED-3Classes_Nested',  'MRI-EQUAL'], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--image_modality", choices=['T1', 'T2', 'EarlyFusion', 'EarlyFusion3Channels', 'LateFusion', 'T1_nopatched', 'T2_nopatched'],
                        help="Which image modality use for MRI.")
    parser.add_argument("--model", choices=["VisionTransformer", "DenseNet", "CNN", "AlexNet", 'ResNet18', 'EfficientNet_b5', 'MobileNet_v2', 'CoAtNet_0', 'CoAtNet_1', 'CoAtNet_2', 'CoAtNet_3', 'CoAtNet_4'], default="DenseNet",
                        help="Model to use.")
    parser.add_argument("--inner_loop_idx", type=int, default= 8,
                        help="Number of fold of inner loop, for Nested train.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16", 
                                                 "ViT-MRI", 'R50-ViT-MRI'],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained", action='store_true',
                        help="If use a pretrained model.")
    parser.add_argument("--split_path", type=str, default="data/10fold_split.json",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--train_only_classifier", action='store_true',
                        help="Freeze all network and train only the classifier")
    parser.add_argument("--accuracy", choices=['simple','balanced', 'both'], default='both', 
                        help="The type of accuracy computed")
    parser.add_argument("--num_fold", choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], type=int, required=True,
                        help="Which Cross Validation folder use.")
    parser.add_argument("--img_size", default=192, type=int,
                        help="Resolution size")
    parser.add_argument("--num_patches", default=9, type=int,
                        help="Number of patches to create a patched input")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--test_batch_size", default=1, type=int,
                        help="Total batch size for test.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--optimizer", choices=["SGD", "Adam", 'RMSprop'], default='SGD', type=str,
                        help="The optimizer.")
    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=2e-7, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--cuda_id", type=int, default=0,
                        help="Index of GPU")
    
    args = parser.parse_args()
    
    timestamp_str = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    args.name = timestamp_str + args.name + '_fold' + str(args.num_fold)
    
    args.output_dir = os.path.join(args.output_dir, args.name)
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    args.loss_weights = get_loss_weights(args.dataset)
    
    # Setup CUDA, GPU & distributed training
    device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    if args.image_modality in  ['T1', 'T1_nopatched']:
        KEYS = ('image_T1', 'label')
    elif args.image_modality in  ['T2', 'T2_nopatched']:
        KEYS = ('image_T2', 'label')
    elif args.image_modality in ['EarlyFusion', 'EarlyFusion3Channels']:
        KEYS = ('fusedImage', 'label')
    elif args.image_modality in ['LateFusion']:
        KEYS = ('image_T1', 'image_T2', 'label')

    
    
    args, model, config = setup(args)
    
    
    
    dict_args = vars(args).copy()
    dict_args['device'] = str(dict_args['device'])
    dict_args['loss_weights'] = dict_args['loss_weights'].tolist()
    # Saving training info
    info = {
        'model_name': model.__class__.__name__,
        'KEYS': KEYS,
        'model_config': config.to_dict(),
        'model_args': dict_args
        }
    with open(os.path.join(args.output_dir, args.name+'.json'), 'w') as fp:
        json.dump(info, fp)

    train(args, logger, model, KEYS, info)    

    
        
if __name__ == "__main__":
    main() 

 