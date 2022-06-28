# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
#from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from torchvision import models

import models.configs as configs
from models.coatnet import CoAtNet

from models.modeling_resnet import ResNetV2


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor, in_channels = in_channels)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        #cls_tokens = cls_tokens.to(x.device)
        if self.hybrid:
            x = self.hybrid_model(x)
        #self.patch_embeddings = self.patch_embeddings.to(x.device)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings#.to(x.device)
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            #query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            query_weight = np2th(weights[ROOT +'/'+ ATTENTION_Q + '/' +"kernel"]).view(self.hidden_size, self.hidden_size).t()
            #key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[ROOT + '/' + ATTENTION_K+ '/' +"kernel"]).view(self.hidden_size, self.hidden_size).t()
            #value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[ROOT + '/' + ATTENTION_V + '/' +"kernel"]).view(self.hidden_size, self.hidden_size).t()
            #out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[ROOT+ '/' + ATTENTION_OUT+ '/' +"kernel"]).view(self.hidden_size, self.hidden_size).t()

            #query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            query_bias = np2th(weights[ROOT + '/' + ATTENTION_Q + '/' + "bias"]).view(-1)
            #key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            key_bias = np2th(weights[ROOT+'/'+ ATTENTION_K +'/'+ "bias"]).view(-1)
            #value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            value_bias = np2th(weights[ROOT +'/'+ ATTENTION_V +'/'+ "bias"]).view(-1)
            #out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)
            out_bias = np2th(weights[ROOT +'/'+ ATTENTION_OUT +'/'+ "bias"]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            #mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_0 = np2th(weights[ROOT+'/'+ FC_0 +'/'+ "kernel"]).t()
            #mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_weight_1 = np2th(weights[ROOT+'/'+ FC_1+'/' + "kernel"]).t()
            #mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_0 = np2th(weights[ROOT +'/'+ FC_0 +'/'+ "bias"]).t()
            #mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()
            mlp_bias_1 = np2th(weights[ROOT +'/'+ FC_1 +'/'+ "bias"]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            #self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.weight.copy_(np2th(weights[ROOT +'/'+ ATTENTION_NORM +'/'+ "scale"]))
            #self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.attention_norm.bias.copy_(np2th(weights[ROOT+'/'+ ATTENTION_NORM+'/'+ "bias"]))
            #self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.weight.copy_(np2th(weights[ROOT+'/'+ MLP_NORM+'/'+ "scale"]))
            #self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
            self.ffn_norm.bias.copy_(np2th(weights[ROOT+'/'+ MLP_NORM+'/'+ "bias"]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, in_channels, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size, in_channels=in_channels)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, in_channels=3, num_classes=21843, loss_weights=None, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.in_channels = in_channels
        self.transformer = Transformer(config, img_size, in_channels, vis)
        self.head = Linear(config.hidden_size, self.num_classes)
        
        if 'grid' not in config.patches.keys():
            self.patch_size = config.patches.size
        else:
            self.patch_size = None
        self.loss_weights = loss_weights

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss(weight = self.loss_weights.to(x.device) if torch.is_tensor(self.loss_weights) else None)
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return logits, loss
        else:
            return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
                  
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())
                
            new_weights = np2th(weights["embedding/kernel"], conv=True)
            if self.in_channels == 1 :
                new_weights = new_weights.mean(dim = 1, keepdim = True)
            elif self.in_channels == 2:
                new_weights = new_weights.mean(dim = 1, keepdim = True)
                new_weights = torch.cat([new_weights, new_weights], dim = 1)
                
            if self.patch_size is not None and self.patch_size != (16,16):
                scale_factor = self.patch_size[0] / 16
                new_weights = torch.nn.functional.interpolate(new_weights,scale_factor = scale_factor, mode = 'bilinear')

            self.transformer.embeddings.patch_embeddings.weight.copy_(new_weights)
            
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)
                        
class LateFusionVisionTransformer(nn.Module):
    
    def __init__(self, config, img_size=224, in_channels=3, num_classes=21843, loss_weights=None, zero_head=False, vis=False):
        super(LateFusionVisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.in_channels = in_channels
        self.transformer_t1 = Transformer(config, img_size, in_channels, vis)
        self.transformer_t2 = Transformer(config, img_size, in_channels, vis)
        self.head = Linear(config.hidden_size*2, self.num_classes)
           
        if 'grid' not in config.patches.keys():
            self.patch_size = config.patches.size
        else:
            self.patch_size = None
        self.loss_weights = loss_weights

    def forward(self, inp, labels=None):
        x_t1, x_t2 = inp
        x_t1, attn_weights = self.transformer_t1(x_t1)
        x_t2, attn_weights = self.transformer_t2(x_t2)
        x = torch.cat((x_t1, x_t2), dim = 2)
        logits = self.head(x[:, 0])
        
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight = self.loss_weights.to(x.device) if torch.is_tensor(self.loss_weights) else None)
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return logits, loss
        else:
            return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
                  
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())
                
            new_weights = np2th(weights["embedding/kernel"], conv=True)
            if self.in_channels == 1 :
                new_weights = new_weights.mean(dim = 1, keepdim = True)
            elif self.in_channels == 2:
                new_weights = new_weights.mean(dim = 1, keepdim = True)
                new_weights = torch.cat([new_weights, new_weights], dim = 1)
                
            if self.patch_size is not None and self.patch_size != (16,16):
                scale_factor = self.patch_size[0] / 16
                new_weights = torch.nn.functional.interpolate(new_weights,scale_factor = scale_factor, mode = 'bilinear')

            self.transformer_t1.embeddings.patch_embeddings.weight.copy_(new_weights)
            
            self.transformer_t1.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer_t1.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer_t1.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer_t1.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            
            self.transformer_t2.embeddings.patch_embeddings.weight.copy_(new_weights)
            
            self.transformer_t2.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer_t2.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer_t2.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer_t2.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer_t1.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer_t1.embeddings.position_embeddings.copy_(posemb)
                self.transformer_t2.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer_t1.embeddings.position_embeddings.copy_(np2th(posemb))
                self.transformer_t2.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer_t1.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)
                    
            for bname, block in self.transformer_t2.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer_t1.embeddings.hybrid:
                self.transformer_t1.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                self.transformer_t2.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer_t1.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer_t1.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)
                
                self.transformer_t2.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer_t2.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer_t1.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)
                        
                for bname, block in self.transformer_t2.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)

class CNNClassifier(nn.Module):
    def __init__(self, in_channels =  3, num_classes = 1000, loss_weights = None, model_type = 'DenseNet', pretrained = False, img_size = None):
        super(CNNClassifier, self).__init__()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.in_channels = in_channels
        
        if model_type== 'DenseNet':
            self.model = models.densenet121(pretrained = pretrained)
            #adapt input and output of densenet for out network parameter
            conv0_old = self.model.features.conv0
            conv_new = torch.nn.Conv2d(in_channels = in_channels, out_channels = conv0_old.out_channels, kernel_size = conv0_old.kernel_size, stride = conv0_old.stride, padding = conv0_old.padding, bias = conv0_old.bias)
            
            self.model.classifier = torch.nn.Linear(in_features = self.model.classifier.in_features, out_features = num_classes )
            
        elif model_type == 'AlexNet':
            self.model = models.alexnet(pretrained = pretrained)
            conv0_old = self.model.features[0]
            conv_new = torch.nn.Conv2d(in_channels, out_channels = conv0_old.out_channels, kernel_size = conv0_old.kernel_size, stride = conv0_old.stride, padding = conv0_old.padding, bias = True)
            
            self.model.classifier.add_module('last_conv',torch.nn.Linear(in_features = self.model.classifier[-1].out_features, out_features = num_classes, bias = True))
            
        elif model_type == 'ResNet18':
            self.model = models.resnet18(pretrained = pretrained)
            conv0_old = self.model.conv1
            conv_new = torch.nn.Conv2d(in_channels, out_channels = conv0_old.out_channels, kernel_size = conv0_old.kernel_size, stride = conv0_old.stride, padding = conv0_old.padding, bias = True)
            
            self.model.fc = torch.nn.Linear(in_features = self.model.fc.in_features, out_features = num_classes, bias = True if self.model.fc.bias is not None else False)
        elif model_type in ['EfficientNet_b5', 'MobileNet_v2']:
            if model_type == 'MobileNet_v2':
                self.model = models.mobilenet_v2(pretrained = pretrained)
            else:
                self.model = models.efficientnet_b5(pretrained = pretrained)
            conv0_old = self.model.features[0][0]
            conv_new = torch.nn.Conv2d(in_channels, out_channels = conv0_old.out_channels, kernel_size = conv0_old.kernel_size, stride = conv0_old.stride, padding = conv0_old.padding, bias = True)
            
            self.model.classifier[1] = torch.nn.Linear(in_features = self.model.classifier[1].in_features, out_features = num_classes, bias = True if self.model.classifier[1].bias is not None else False)
            
        elif 'CoAtNet' in model_type:
            if '0' in model_type:
                num_blocks = [2, 2, 3, 5, 2]            # L
                channels = [64, 96, 192, 384, 768]      # D
                block_types = ['C', 'C', 'T', 'T']
            elif '1' in model_type:
                num_blocks = [2, 2, 6, 14, 2]
                channels = [64, 96, 192, 384, 768]
                block_types = ['C', 'C', 'T', 'T']
            elif '2' in model_type:
                num_blocks = [2, 2, 6, 14, 2]
                channels = [128, 128, 256, 512, 1026]
                block_types = ['C', 'C', 'T', 'T']
            elif '3' in model_type:
                num_blocks = [2, 2, 6, 14, 2]
                channels = [192, 192, 384, 768, 1536]
                block_types = ['C', 'C', 'T', 'T']
            elif '4' in model_type:
                num_blocks = [2, 2, 12, 28, 2]
                channels = [192, 192, 384, 768, 1536]
                block_types = ['C', 'C', 'T', 'T']
            elif '5' in model_type: #changing DIM head from 32 to 64
                num_blocks = [2, 2, 12, 28, 2]
                channels = [192, 256, 512, 1280, 2048]
                block_types = ['C', 'C', 'T', 'T']
            self.model = CoAtNet((img_size, img_size), in_channels, num_blocks, channels, num_classes, block_types = block_types)
            conv0_old = self.model.s0[0][0]
            conv_new = torch.nn.Conv2d(in_channels, out_channels = conv0_old.out_channels, kernel_size = conv0_old.kernel_size, stride = conv0_old.stride, padding = conv0_old.padding, bias = True)
            
            self.model.fc = nn.Linear(in_features = self.model.fc.in_features, out_features=num_classes, bias = True if self.model.fc.bias is not None else False)
            
        if pretrained:
            w_new = conv0_old.weight.data
            if in_channels == 1 or in_channels==2:
                w_new = w_new.mean(dim = 1, keepdims = True)
                if in_channels==2:
                    w_new = torch.cat((w_new,w_new), dim=1)
            conv_new.weight.data = w_new   
            
        if model_type == 'DenseNet':
            self.model.features.conv0 = conv_new
        elif model_type == 'AlexNet':
            self.model.features[0] = conv_new
        elif model_type == 'ResNet18':
            self.model.conv1 = conv_new
        elif model_type in ['EfficientNet_b5', 'MobileNet_v2']:
            self.model.features[0][0] = conv_new
        elif 'CoAtNet' in model_type:
            self.model.s0[0][0] = conv_new
            
    def forward(self, x, labels = None):
        
        logits = self.model(x)
        
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight = self.loss_weights.to(x.device) if torch.is_tensor(self.loss_weights) else None)
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return logits, loss
        
        return logits, None
    
class CNN(nn.Module):
    
    def __init__(self, in_channels, num_classes):
        super().__init__();
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=8, padding=0, stride=8),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            # Classification layer
            nn.Linear(1024, num_classes)
        )

    # Forward
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-MRI' : configs.get_r50_MRI_config(),
    'ViT-MRI' : configs.get_MRI_config(),
    'testing': configs.get_testing(),
    'None':  configs.get_none()
}
