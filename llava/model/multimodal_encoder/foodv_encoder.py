import torch
import torch.nn as nn

from llava.model.multimodal_encoder.foodv_model import foodvImageEncoder, foodvImageProcessor
import os
import argparse
import re
from typing import List, Optional, Union
import sys

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 project 目录
project_dir = os.path.abspath(os.path.join(current_dir, '../../../../'))
# 获取 corenet1 目录
corenet1_dir = os.path.join(project_dir, 'corenet')
# 将 corenet1 目录添加到 sys.path 中
sys.path.append(corenet1_dir)
from .load_pretrain_model import load_pretrained_model
from llava.model.multimodal_encoder.foodv_config import get_configuration
from corenet.utils.common_utils import unwrap_model_fn


class FoodVisionTower(nn.Module):
    def __init__(self, image_tower, args, delay_load=False, cache_dir='./cache_dir', dtype=torch.bfloat16):
        super().__init__()

        self.is_loaded = False
        #self.pretrain_path = args.pretrain_path
        self.pretrain_path = '/media/fast_data/model/foodv_base_19.pt'
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.args = args
        self.cache_dir = cache_dir
        self.config = get_configuration(args)
        self.model = foodvImageEncoder(self.config)
        self.hidden_size = self.model.block1_embed_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.load_model()
        
    def load_model(self):
        self.image_processor = foodvImageProcessor(image_size=224)
        self.image_tower = load_pretrained_model(self.model, self.pretrain_path, self.device, self.args)
        self.image_tower.requires_grad_(False)
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                x = self.image_tower(image.to(device=self.device, dtype=self.dtype))
                image_features.append(x)
        else:
            image_features = self.image_tower(images.to(device=self.device, dtype=self.dtype))

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

