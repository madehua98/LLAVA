#!/bin/bash

export PYTHONPATH=/home/data_llm/madehua/LLaVA
cd /home/data_llm/madehua/LLaVA

CUDA_VISIBLE_DEVICES=3 python llava/eval/model_vqa.py \
    --model-path /media/fast_data/model/llava-v1.5-7b \
    --question-file \
    /mnt/data_llm/json_file/101_questions_noft_v1.jsonl.jsonl \
    --image-folder \
    /media/fast_data \
    --answers-file \
    /mnt/data_llm/json_file/llava1.5-7b.jsonl