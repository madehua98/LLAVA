import torch
import torch.nn as nn

import os
import argparse
import re
from typing import List, Optional, Union
import sys
from collections import OrderedDict
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 project 目录
project_dir = os.path.abspath(os.path.join(current_dir, '../../../../'))
# 获取 corenet1 目录
corenet1_dir = os.path.join(project_dir, 'corenet')
# 将 corenet1 目录添加到 sys.path 中
sys.path.append(corenet1_dir)

from corenet.utils.common_utils import unwrap_model_fn

def clean_strip(
    obj: Union[str, List[str]], sep: Optional[str] = ",", strip: bool = True
) -> List[str]:
    if isinstance(obj, list):
        strings = obj
    else:
        strings = obj.split(sep)

    if strip:
        strings = [x.strip() for x in strings]
    strings = [x for x in strings if x]
    return strings

def initialize_meta_parameters(model, device):
    """Ensure all meta parameters and buffers are initialized with actual data before moving to the device."""
    meta_params = []
    meta_buffers = []

    # 收集所有需要修改的参数和缓冲区
    for name, param in model.named_parameters():
        if param.is_meta:
            meta_params.append((name, param))

    for name, buffer in model.named_buffers():
        if buffer.is_meta:
            meta_buffers.append((name, buffer))

    # 逐级处理元参数
    for name, param in meta_params:
        print(f"{name} is a meta tensor.")
        new_param = torch.nn.Parameter(torch.zeros(param.shape, device=device))
        module_names = name.split('.')[:-1]
        param_name = name.split('.')[-1]
        module = model
        for module_name in module_names:
            module = getattr(module, module_name)
        # 先删除旧参数，再设置新参数
        del module._parameters[param_name]
        module._parameters[param_name] = new_param

    # 逐级处理元缓冲区
    for name, buffer in meta_buffers:
        print(f"{name} is a meta tensor.")
        new_buffer = torch.zeros(buffer.shape, device=device)
        module_names = name.split('.')[:-1]
        buffer_name = name.split('.')[-1]
        module = model
        for module_name in module_names:
            module = getattr(module, module_name)
        # 先删除旧缓冲区，再设置新缓冲区
        del module._buffers[buffer_name]
        module._buffers[buffer_name] = new_buffer


def convert_to_bfloat16(model):
    """Convert model parameters and buffers to bfloat16."""
    for param in model.parameters():
        param.data = param.data.to(torch.bfloat16)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(torch.bfloat16)
        
        
def load_pretrained_model(
    model: torch.nn.Module, wt_loc: str, device: torch.device, *args, **kwargs
) -> torch.nn.Module:
    if isinstance(wt_loc, OrderedDict):
        wts = wt_loc
    else:
        wts = torch.load(wt_loc, map_location="cpu")
    print("Loaded weights keys: ", wts.keys())
    
    if not args:
        raise ValueError("Expected additional arguments in 'args'")
    args = args[0]

    exclude_scopes = getattr(args, 'resume_exclude_scopes', [])
    exclude_scopes = clean_strip(exclude_scopes)
    missing_scopes = getattr(args, 'ignore_missing_scopes', ['post_transformer_norm.running_mean', 'post_transformer_norm.running_var'])
    missing_scopes = clean_strip(missing_scopes)

    rename_scopes_map = getattr(args, 'rename_scopes_map', [])
    if rename_scopes_map:
        for entry in rename_scopes_map:
            if len(entry) != 2:
                raise ValueError(
                    "Every entry in model.rename_scopes_map must contain exactly two string elements"
                    " for before and after. Got {}.".format(str(entry))
                )
    exclude_scopes += [
        'classifier',
        'neural_augmentor.brightness._low',
        'neural_augmentor.brightness._high',
        'neural_augmentor.contrast._low',
        'neural_augmentor.contrast._high',
        'neural_augmentor.noise._low',
        'neural_augmentor.noise._high',
        'patch_emb.0.block.norm.weight',
        'patch_emb.0.block.norm.bias',
        'patch_emb.0.block.norm.running_mean',
        'patch_emb.0.block.norm.running_var',
        'patch_emb.0.block.norm.num_batches_tracked',
        'patch_emb.1.block.norm.weight',
        'patch_emb.1.block.norm.bias',
        'patch_emb.1.block.norm.running_mean',
        'patch_emb.1.block.norm.running_var',
        'patch_emb.1.block.norm.num_batches_tracked',
    ]
    missing_scopes += exclude_scopes

    if exclude_scopes:
        for key in list(wts.keys()):
            if any(re.match(x, key) for x in exclude_scopes):
                del wts[key]

    if rename_scopes_map:
        for before, after in rename_scopes_map:
            wts = {re.sub(before, after, key): value for key, value in wts.items()}
    
    print("Modified weights keys: ", wts.keys())

    strict = not bool(missing_scopes)

    module = unwrap_model_fn(model)
    
    print("Model before loading state dict: ", module)
    print("Type of module: ", type(module))
    
    if not isinstance(module, torch.nn.Module):
        raise TypeError("The unwrapped model is not an instance of torch.nn.Module")
    
    initialize_meta_parameters(module, device)
    
    try:
        missing_keys, unexpected_keys = module.load_state_dict(wts, strict=strict)
    except TypeError as e:
        print(f"Error loading state dict: {e}")
        raise

    print("Missing keys: ", missing_keys)
    print("Unexpected keys: ", unexpected_keys)

    if unexpected_keys:
        raise Exception(
            "Found unexpected keys: {}."
            " You can ignore these keys using `model.resume_exclude_scopes`.".format(
                ", ".join(unexpected_keys)
            )
        )

    missing_keys = [
        key
        for key in missing_keys
        if not any(re.match(x, key) for x in missing_scopes)
    ]

    if missing_keys:
        raise Exception(
            "Missing keys detected. Did not find the following keys in pre-trained model: {}."
            " You can ignore the keys using `model.ignore_missing_scopes`.".format(
                ",".join(missing_keys)
            )
        )

    model.to(device)

    convert_to_bfloat16(model)
    return model