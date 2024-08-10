import json
import random
import shutil
import os

import json
import random
import shutil
import os
from tqdm import tqdm

def extract_sample_images(jsonl_file_path, output_json_path, sample_size=100000, prompts=None):
    all_data = []
    result_data = []

    count = 0
    with open(jsonl_file_path, "r") as file:
        line_number = 0
        for line in file:
            line_number += 1
            try:
                data = json.loads(line)
                all_data.append(data)
            except json.JSONDecodeError as e:
                count += 1
                #print(f"Error loading json on line {line_number}: {str(e)}")
    print(count)
    sample_data = random.sample(all_data, sample_size)

    for data in tqdm(sample_data, total=len(sample_data)):
        # 处理和复制图像文件
        original_image_path = data['image']
        new_image_path = original_image_path.replace('image', 'sample_image')
        new_image_relpath = '/'.join(new_image_path.split('/')[-7:])
        
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
        shutil.copy2(original_image_path, new_image_path)

        # 创建新的 JSON 对象
        title_conversation = [
            {"from": "human", "value": f"<image>\n{random.choice(title_prompts)}"},
            {"from": "gpt", "value": data['title']}
        ]
        title_json_object = {
            "image": new_image_relpath,
            "conversations": title_conversation
        }
        result_data.append(title_json_object)
        ingredient_conversation = [
            {"from": "human", "value": f"<image>\n{random.choice(ingredient_prompts)}"},
            {"from": "gpt", "value": data['ingredient']}
        ]
        ingredient_json_object = {
            "image": new_image_relpath,
            "conversations": ingredient_conversation
        }
        result_data.append(ingredient_json_object)
        recipe_conversation = [
            {"from": "human", "value": f"<image>\n{random.choice(recipe_prompts)}"},
            {"from": "gpt", "value": data['instructions']}
        ]
        recipe_json_object = {
            "image": new_image_relpath,
            "conversations": recipe_conversation
        }
        result_data.append(recipe_json_object)

    # 写入 JSON 数据到文件
    with open(output_json_path, 'w') as outfile:
        json.dump(result_data, outfile, indent=4)

# prompts = [
#     'Render a clear and concise summary of the photo.',
#     '"Write a terse but informative summary of the picture.',
#     'What is this?',
#     'What is in the photo?',
#     'Describe the image concisely.',
#     'Share a concise interpretation of the image provided.',
#     'Give a brief description of the image.',
#     "Present a compact description of the photo's key features.",
#     'Share a concise interpretation of the image provided.',
#     'Provide a brief description of the given image.',
#     'What is in the photo?',
#     'Summarize the visual content of the image.'
# ]

ingredient_prompts = [
    "Can you identify the ingredients present in this image?",
    "What ingredients are visible in this picture?",
    "Can you identify the ingredients present in this image?",
    "Which food ingredients can you discern in this photo?",
    "Can you identify the ingredients from this picture?",
    "What are the ingredients of the dish depicted in the image?",
    "Can you list the components of the dish shown in the photo?",
    "What is the dish in the picture made of?",
]

title_prompts = [
    "What is the name of the dish shown in the image?",
    "What is the name of this dish?",
    "What is the name of the dish shown in the image?",
    "Can you tell me the dish's name?",
    "What dish is this?",
    "Can you tell me the name of this dish?",
    "What is the culinary name of this dish?",
    "Can you provide the name of the dish?",
    "What is the category of the dish presented in the image?",
    "Can you identify the dish displayed in the photo?",
    "Which dish is depicted in the picture?",
]

recipe_prompts = [
    "Can you provide a recipe or guide to recreate this dish?",
    "Could you please provide a step-by-step recipe for this dish?",
    "Could you explain the cooking process for this dish in detail",
    "How to make this dish?",
    "What are the steps or instructions for making the dish in the image?",
    "Please provide a detailed guide on how to prepare this dish.",
    "How do I cook the dish in this image?",
    "What's the recipe for the dish in this image?",
    "What's the step-by-step process to recreate this dish from the image?",
    "Can you break down the cooking steps shown in this photo?",
]

# JSONL 文件路径和输出 JSON 文件路径
# 指定 JSONL 文件路径
jsonl_file_path = '/media/fast_data/recipe1M+/image2texts_new.jsonl'
output_json_path = '/media/fast_data/recipe1M+/recipe_llava_pretrain.json'

# 调用函数
# extract_sample_images(jsonl_file_path, output_json_path, 70000, recipe_prompts)


input_json1 = '/media/fast_data/datacomp_sample/datacomp_1b_new.json'
input_json2 = '/media/fast_data/laion2b_sample/laion2b_new.json'
input_json3 = '/media/fast_data/recipe1M+/recipe_llava_pretrain.json'
new_datas = []
with open(input_json1, 'r') as f:
    datas1 = json.load(f)

with open(input_json2, 'r') as f:
    datas2 = json.load(f)
    
with open(input_json3, 'r') as f:
    datas3 = json.load(f)

new_datas = []
new_datas.extend(datas1)
new_datas.extend(datas2)
new_datas.extend(datas3)
random.shuffle(new_datas)
# for data in datas:
#     new_data = data
#     new_data["image"] = data["image"].replace("sample_extracted_shards", "datacomp_sample/sample_extracted_shards")
#     new_datas.append(new_data)

output_json = '/mnt/data_llm/json_file/llava_pretrain_data.json'
with open(output_json, 'w') as file:
    json.dump(new_datas, file, indent=4)