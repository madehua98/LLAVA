import torch
import torch.nn as nn

from llava.model.multimodal_encoder.vit_model import VitImageEncoder, VitImageProcessor
import os
import argparse
import re
from typing import List, Optional, Union
import sys
from llava.model.multimodal_encoder.load_pretrain_model import load_pretrained_model

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 project 目录
project_dir = os.path.abspath(os.path.join(current_dir, '../../../../'))
# 获取 corenet1 目录
corenet1_dir = os.path.join(project_dir, 'corenet')
# 将 corenet1 目录添加到 sys.path 中
sys.path.append(corenet1_dir)

from llava.model.multimodal_encoder.vit_config import get_configuration
from torch.cuda.amp import autocast, GradScaler


class VitVisionTower(nn.Module):
    def __init__(self, image_tower, args, delay_load=False, cache_dir='./cache_dir', dtype=torch.bfloat16):
        super().__init__()

        self.is_loaded = False
        #self.pretrain_path = args.pretrain_path
        self.pretrain_path = '/media/fast_data/model/catlip_vit_base.pt'
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.args = args
        self.cache_dir = cache_dir
        self.config = get_configuration(args)
        self.model = VitImageEncoder(self.config)
        self.model.neural_augmentor = None
        self.hidden_size = self.model.embed_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.load_model()
    
    def rename_model_weights(self):
        new_state_dict = {}
        for name, param in self.model.state_dict().items():
            # 对于每一层的pre_norm_mha.1.out_proj.weight和pre_norm_mha.1.out_proj.bias进行修改
            for i in range(len(self.model.transformer)):
                target_weight_str = f'transformer.{i}.pre_norm_mha.1.out_proj.weight'
                new_weight_str = f'transformer.{i}.pre_norm_mha.1.out_proj_attn.weight'
                target_bias_str = f'transformer.{i}.pre_norm_mha.1.out_proj.bias'
                new_bias_str = f'transformer.{i}.pre_norm_mha.1.out_proj_attn.bias'

                if target_weight_str in name:
                    new_name = name.replace(target_weight_str, new_weight_str)
                    new_state_dict[new_name] = param
                elif target_bias_str in name:
                    new_name = name.replace(target_bias_str, new_bias_str)
                    new_state_dict[new_name] = param
                else:
                    new_state_dict[name] = param

        return new_state_dict
    
    def rename_pretrain_model(self):
        dict = torch.load(self.pretrain_path)
        items = list(dict.items())  # 将字典的键值对存储到一个临时列表中

        for name, param in items:
            # 对于每一层的 pre_norm_mha.1.out_proj.weight 和 pre_norm_mha.1.out_proj.bias 进行修改
            for i in range(len(self.model.transformer)):
                target_weight_str = f'transformer.{i}.pre_norm_mha.1.out_proj.weight'
                new_weight_str = f'transformer.{i}.pre_norm_mha.1.out_proj_attn.weight'
                target_bias_str = f'transformer.{i}.pre_norm_mha.1.out_proj.bias'
                new_bias_str = f'transformer.{i}.pre_norm_mha.1.out_proj_attn.bias'
                if new_weight_str in name:
                    dict[target_weight_str] = dict.pop(new_weight_str)
                elif new_bias_str in name:
                    dict[target_bias_str] = dict.pop(new_bias_str)
                    
        return dict
    
    def convert_to_bfloat16(self, model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):  # BatchNorm layers retain original dtype
                continue
            for param in module.parameters():
                param.data = param.data.to(torch.bfloat16)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(torch.bfloat16)
    def load_model(self):
        self.image_processor = VitImageProcessor(image_size=224)
        dict = self.rename_pretrain_model()
        self.model.ignore_missing_scopes = ['post_transformer_norm.running_mean', 'post_transformer_norm.running_var']
        self.image_tower = load_pretrained_model(self.model, dict, self.device, self.args)
        self.image_tower.requires_grad_(False)
        self.is_loaded = True
        self.convert_to_bfloat16(self.image_tower)
        

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                out_dict = self.image_tower(image.to(device=self.device, dtype=self.dtype))
                all_hidden_states = out_dict["hidden_states"]
                image_feature = all_hidden_states[self.select_layer].to(self.dtype)
                image_features.append(image_feature)
        else:
            with autocast():
                out_dict = self.image_tower(images.to(device=self.device, dtype=self.dtype))
                all_hidden_states = out_dict["hidden_states"]
                image_features = all_hidden_states[self.select_layer].to(self.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



# def rename_pretrain_model(pretrain_path):
#     dict = torch.load(pretrain_path)
#     for name, param in dict.items():
#         # 对于每一层的pre_norm_mha.1.out_proj.weight和pre_norm_mha.1.out_proj.bias进行修改
#         for i in range(12):
#             target_weight_str = f'transformer.{i}.pre_norm_mha.1.out_proj.weight'
#             new_weight_str = f'transformer.{i}.pre_norm_mha.1.out_proj_attn.weight'
#             target_bias_str = f'transformer.{i}.pre_norm_mha.1.out_proj.bias'
#             new_bias_str = f'transformer.{i}.pre_norm_mha.1.out_proj_attn.bias'
#             if new_weight_str in name:
#                 dict[target_weight_str] = dict.pop(new_weight_str)
#             elif new_bias_str in name:
#                 dict[target_bias_str] = dict.pop(new_bias_str)
#     return dict

# path = '/media/fast_data/model/catlip_vit_base.pt'
# model1 = rename_pretrain_model(path)
# print(model1.keys())