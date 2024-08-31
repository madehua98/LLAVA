#!/bin/bash

export PYTHONPATH=/home/data_llm/madehua/LLaVA
cd /home/data_llm/madehua/LLaVA

CUDA_VISIBLE_DEVICES=1 python llava/eval/model_vqa.py \
    --model-path /mnt/data_llm/model/checkpoints/llava1.5-7b-vitl-nofood-food101/checkpoint-1184 \
    --question-file \
    /mnt/data_llm/json_file/101_questions.jsonl \
    --image-folder \
    /media/fast_data \
    --answers-file \
    /mnt/data_llm/json_file/llava1.5-7b-vitl-nofood-1184-101.jsonl