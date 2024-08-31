import json
import shutil
path = '/mnt/data_llm/json_file/nutrition5k_train_modified2.json'
path1 = '/mnt/data_llm/json_file/nutrition5k_test_modified2.json'
import os
def read_json(filename):
    with open(filename, 'r') as file:
        datas = json.load(file)
    return datas

def copy_files(data_list):
    for data in data_list:
        original_path = data['image']
        new_path = original_path.replace('imagery', 'sample_imagery')
        new_directory = os.path.dirname(new_path)
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)  # 创建所有必要的父目录
        shutil.copy2(original_path, new_path)  # copy2 also copies the metadata
        print(f"File copied from {original_path} to {new_path}")
        # Checking if original file exists and then copying
        # try:
        #     shutil.copy2(original_path, new_path)  # copy2 also copies the metadata
        #     print(f"File copied from {original_path} to {new_path}")
        # except FileNotFoundError:
        #     print(f"File not found: {original_path}")
        # except Exception as e:
        #     print(f"An error occurred: {str(e)}")

train_datas = read_json(path)
test_datas = read_json(path1)

copy_files(train_datas)
copy_files(test_datas)
