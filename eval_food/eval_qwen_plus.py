from http import HTTPStatus
import dashscope
import json
import os
from tqdm import tqdm
import time

dashscope.api_key = 'sk-9fb9a9739b864006b52a6fe7f9cf3985'

def load_jsonl(file_path):
    try:
        results = []
        with open(file_path, 'r') as file:
            for line in file:
                line = json.loads(line)
                results.append(line)
        return results
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
        return []

def save_jsonl(file_path, data):
    try:
        with open(file_path, 'a') as file:
            file.write(json.dumps(data, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error saving to JSONL file: {e}")

def simple_multimodal_conversation_call(model, image_path):
    """Simple single round multimodal conversation call."""
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_path},
                {"text": "What dish is this? Keep output as short as possible"}
            ]
        }
    ]
    try:
        response = dashscope.MultiModalConversation.call(model=model, messages=messages)
        if response.status_code == HTTPStatus.OK:
            try:
                category = response['output']['choices'][0]['message']['content'][0]['text']
                return category
            except (KeyError, IndexError, TypeError):
                print("Error: Unexpected response structure.")
                return None
        else:
            print(f"Error code: {response.code}")
            print(f"Error message: {response.message}")
            return None
    except Exception as e:
        print(f"Error in API call: {e}")
        return None

def retry_call_simple_multimodal(model, image_path, retries=10):
    """Retry the simple_multimodal_conversation_call up to `retries` times if None is returned."""
    for attempt in range(retries):
        category = simple_multimodal_conversation_call(model, image_path)
        if category is not None:
            return category
        print(f"Retry {attempt + 1}/{retries} failed. Retrying...")
    print("Max retries reached. Returning None.")
    return None

if __name__ == '__main__':
    questions_jsonl = '/mnt/data_llm/json_file/101_questions.jsonl'
    answers_jsonl = '/mnt/data_llm/json_file/101-qwen-vl-plus.jsonl'
    questions = load_jsonl(questions_jsonl)
    model = 'qwen-vl-plus'
    
    # 使用 tqdm 包装问题列表以显示进度条
    for question in tqdm(questions, desc="Processing questions"):
        image = question.get('image')
        question_id = question.get('question_id')
        
        if image and question_id:
            category = retry_call_simple_multimodal(model, image)
            messages = {
                'question_id': question_id,
                'image': image,
                'text': category,
                'category': 'default'
            }
            save_jsonl(answers_jsonl, messages)
        else:
            print(f"Invalid question data: {question}")