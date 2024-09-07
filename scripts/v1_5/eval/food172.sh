#!/bin/bash

export PYTHONPATH=/home/data_llm/madehua/LLaVA
cd /home/data_llm/madehua/LLaVA

CUDA_VISIBLE_DEVICES=9 python llava/eval/model_vqa.py \
    --model-path /media/fast_data/model/llava-v1.5-7b \
    --question-file \
    /mnt/data_llm/json_file/172_questions_noft_v1.jsonl \
    --image-folder \
    /media/fast_data \
    --answers-file \
    /mnt/data_llm/json_file/llava1.5-7b-172.jsonl