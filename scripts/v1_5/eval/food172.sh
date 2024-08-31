#!/bin/bash

export PYTHONPATH=/home/data_llm/madehua/LLaVA
cd /home/data_llm/madehua/LLaVA

CUDA_VISIBLE_DEVICES=0 python llava/eval/model_vqa.py \
    --model-path /mnt/data_llm/model/checkpoints/llava1.5-7b-vitl-nofood-food172/checkpoint-1032 \
    --question-file \
    /mnt/data_llm/json_file/172_questions.jsonl \
    --image-folder \
    /media/fast_data \
    --answers-file \
    /mnt/data_llm/json_file/llava1.5-7b-vitl-nofood-1032-172.jsonl