#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict
import sys
import os

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 project 目录
project_dir = os.path.abspath(os.path.join(current_dir, '../../../../'))
# 获取 corenet1 目录
corenet1_dir = os.path.join(project_dir, 'corenet')
# 将 corenet1 目录添加到 sys.path 中
sys.path.append(corenet1_dir)

import torch
from corenet.utils import logger
from typing import List, Optional, Union
from torch import Size, Tensor, nn
class LayerNorm(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        *args,
        **kwargs
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )


class LayerNormFP32(LayerNorm):
    """
    Applies `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ over a input tensor with FP32 precision
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        *args,
        **kwargs
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            *args,
            **kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        # Convert input from dtype X to FP32 and perform normalization operation.
        # This may help with underflow/overflow issues that we typically see with normalization layers
        inp_dtype = x.dtype
        return super().forward(x.to(torch.float32)).to(inp_dtype)

def get_configuration(args) -> Dict:
    #mode = args.mode if args.mode is not None else 'small'
    mode = 'base'
    mode = mode.lower()
    if not mode:
        logger.error("Please specify mode")

    mode = mode.lower()
    dropout = getattr(args, "dropout", 0.0)
    norm_layer = LayerNormFP32

    vit_config = dict()
    if mode == "tiny":
        vit_config = {
            "embed_dim": 192,
            "n_transformer_layers": 12,
            "n_attn_heads": 3,
            "ffn_dim": 192 * 4,
            "norm_layer": norm_layer,
            "pos_emb_drop_p": 0.1,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": 0.0,
            "use_cls_token": True
        }
    elif mode == "small":
        vit_config = {
            "embed_dim": 384,
            "n_transformer_layers": 12,
            "n_attn_heads": 6,
            "ffn_dim": 384 * 4,
            "norm_layer": norm_layer,
            "pos_emb_drop_p": 0.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": 0.0,
            "use_cls_token": True
        }
    elif mode == "base":
        vit_config = {
            "embed_dim": 768,
            "n_transformer_layers": 12,
            "n_attn_heads": 12,
            "ffn_dim": 768 * 4,
            "norm_layer": norm_layer,
            "pos_emb_drop_p": 0.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": 0.0,
            "use_cls_token": True
        }
    elif mode == "large":
        vit_config = {
            "embed_dim": 1024,
            "n_transformer_layers": 24,
            "n_attn_heads": 16,
            "ffn_dim": 1024 * 4,
            "norm_layer": norm_layer,
            "pos_emb_drop_p": 0.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": dropout,
        }
    elif mode == "huge":
        vit_config = {
            "embed_dim": 1280,
            "n_transformer_layers": 32,
            "n_attn_heads": 20,  # each head dimension is 64
            "ffn_dim": 1280 * 4,
            "norm_layer": norm_layer,
            "pos_emb_drop_p": 0.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": dropout,
        }
    else:
        logger.error("Got unsupported ViT configuration: {}".format(mode))
    return vit_config
