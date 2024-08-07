#!/bin/bash

export PYTHONPATH=/home/data_llm/madehua/LLaVA
cd /home/data_llm/madehua/LLaVA
gpus="1,2,3,4,5,6,7,8"
echo $gpus
image_tower='/media/fast_data/model/foodv_base_19.pt'

deepspeed --include localhost:$gpus  llava/train/train_xformers.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /media/fast_data/model/vicuna-7b-v1.5 \
    --version plain \
    --data_path /media/fast_data/datacomp_sample/datacomp_1b_new.json \
    --image_folder /media/fast_data \
    --vision_tower $image_tower \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /mnt/data_llm/model/checkpoints/llava1.5-7b-datacomp \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 64 \
    --lazy_preprocess True
    --report_to none
