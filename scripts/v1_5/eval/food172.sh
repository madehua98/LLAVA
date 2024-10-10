#!/bin/bash

export PYTHONPATH=/home/data_llm/madehua/LLaVA
cd /home/data_llm/madehua/LLaVA

CUDA_VISIBLE_DEVICES=2 python llava/eval/model_vqa.py \
    --model-path /mnt/data_llm/model/checkpoints/llava1.5-7b-vitl-retrieval-mix-food101/checkpoint-592 \
    --question-file \
    /mnt/data_llm/json_file/172_questions_retrieval_5.jsonl \
    --image-folder \
    /media/fast_data \
    --answers-file \
    /mnt/data_llm/json_file/llava1.5-7b-raft-101-r5-172.jsonl