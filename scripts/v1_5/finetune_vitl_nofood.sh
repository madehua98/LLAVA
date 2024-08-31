#!/bin/bash

export PYTHONPATH=/home/data_llm/madehua/LLaVA
cd /home/data_llm/madehua/LLaVA
gpus="0,2,3,4,5,6,7,9"
echo $gpus
#image_tower='/media/fast_data/model/catlip_vit_base.pt'
image_tower='/media/fast_data/model/clip-vit-large-patch14-336'

deepspeed --include localhost:$gpus llava/train/train_xformers.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /media/fast_data/model/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /mnt/data_llm/json_file/172_train_prompt10.json \
    --image_folder /media/fast_data \
    --vision_tower $image_tower \
    --pretrain_mm_mlp_adapter /mnt/data_llm/model/llava1.5-7b-vitl-nofood-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /mnt/data_llm/model/checkpoints/llava1.5-7b-vitl-nofood-food172 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 64 \
    --lazy_preprocess True 
    # --report_to None
