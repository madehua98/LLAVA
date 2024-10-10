from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm


def save_jsonl(filename, data):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_jsonl(filename):
    results = []
    with open(filename, 'r') as f:
        for line in f:
            line = json.loads(line)
            results.append(line)
    return results

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

###############################################################################################使用DINOV2进行检索###############################################################################################
# 根目录路径
#root_dir = '/mnt/data_llm/food_images/VireoFood172/ready_chinese_food/'
root_dir = '/media/fast_data/Food2k_complete/'
# root_dir = '/media/fast_data/food_recognition_dataset/food-101/images/'
#root_dir = '/mnt/data_llm/food_images/food200/images/'
# 训练数据和测试数据文件路径
# train_file = '/mnt/data_llm/food_images/VireoFood172/SplitAndIngreLabel/TR.txt'
# test_file = '/mnt/data_llm/food_images/VireoFood172/SplitAndIngreLabel/TE.txt'
train_file = '/media/fast_data/Food2k_complete/train.txt'
test_file = '/media/fast_data/Food2k_complete/test.txt'
# train_file = '/media/fast_data/food_recognition_dataset/food-101/meta/train.txt'
# test_file = '/media/fast_data/food_recognition_dataset/food-101/meta/test.txt'
# train_file = '/mnt/data_llm/food_images/food200/metadata/train_finetune_v2.txt'
# test_file = '/mnt/data_llm/food_images/food200/metadata/test_finetune_v2.txt'

# 自定义Dataset类
class FoodDataset(Dataset):
    def __init__(self, file_path, root_dir, label, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.label = label
        self.image_paths = self.load_image_paths(file_path)
        
    def load_image_paths(self, file_path):
        image_paths = []
        with open(file_path, 'r') as file:
            for line in file:
                relative_path = line.strip()
                relative_path = relative_path.split(' ')
                full_path = self.root_dir + relative_path[0]
                if 'jpg' not in full_path:
                    full_path = full_path + '.jpg'
                image_paths.append(full_path)
        return image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt")
            inputs['label'] = self.label
            inputs['image_path'] = image_path
            return inputs
        except:
            print(f"Error reading item at index {idx}, trying next one.")
            return self.__getitem__(idx + 1)


#model_path = '/media/fast_data/model/dinov2-large'
model_path = '/media/fast_data/model/clip-vit-large-patch14'
# 初始化处理器
processor = AutoImageProcessor.from_pretrained(model_path)
print(processor.size)

# 创建训练和测试数据集
train_dataset = FoodDataset(train_file, root_dir, "train", processor)
test_dataset = FoodDataset(test_file, root_dir, "test", processor)

# 合并数据集
full_dataset = train_dataset + test_dataset

# 定义DataLoader
batch_size = 16
dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=64)

# 初始化模型并放置在GPU上
model = AutoModel.from_pretrained(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
image_encoder = model.vision_model

# 用于存放数据的列表
data_list = []

# 使用tqdm添加进度条
for batch in tqdm(dataloader, desc="Processing images"):
    
    # 获取图像路径和标签
    batch_labels = batch.pop('label')
    batch_paths = batch.pop('image_path')
    batch_pixel_values = batch['pixel_values'].to(device)
    batch_pixel_values = batch_pixel_values.squeeze(1)  # 移除第二维度(1)
    # 使用模型获取输出
    outputs = image_encoder(pixel_values=batch_pixel_values)
    
    # 获取 cls_token_embedding 和 average_embedding
    last_hidden_states = outputs.last_hidden_state
    cls_token_embeddings = last_hidden_states[:, 0, :].detach().cpu().numpy().tolist()  # 转回CPU
    #average_embeddings = torch.mean(last_hidden_states, dim=1).detach().cpu().numpy().tolist()  # 转回CPU
    
    # 为当前批次创建字典并加入到列表中
    for i in range(len(batch_paths)):
        data = {
            'image_path': batch_paths[i],
            'label': batch_labels[i],  # 添加训练/测试标签
            'cls_token_embedding': cls_token_embeddings[i],
            # 'average_embedding': average_embeddings[i]
        }
        data_list.append(data)

# 将数据保存到 JSON 文件
with open('/mnt/data_llm/food2k_embeddings_clip_large.json', 'w') as json_file:
    json.dump(data_list, json_file, indent=4)

print("数据已成功保存到 JSON 文件中")


###############################################################计算每个数据集检索的索引#############################################################
# import os
# import json
# import numpy as np
# from tqdm import tqdm

# def compute_similarity(test_tensors, train_tensors):
#     """
#     计算测试图片张量与训练图片张量的相似度，并返回排序索引
#     :param test_tensors: 测试图片张量的矩阵，形状为 (num_test_images, tensor_dim)
#     :param train_tensors: 训练图片张量的矩阵，形状为 (num_train_images, tensor_dim)
#     :return: 对于每个测试图片，返回其与所有训练图片的相似度排序索引的矩阵，形状为 (num_test_images, num_train_images)
#     """
#     # 计算张量的内积相似度矩阵
#     similarity_matrix = np.dot(test_tensors, train_tensors.T)
    
#     # 对相似度矩阵进行排序，返回从大到小的排序索引
#     sorted_indices = np.argsort(-similarity_matrix, axis=1)
    
#     return sorted_indices

# def get_tensors(data_list, label):
#     tensors = []
#     category_list = []
#     path_list = []
#     for data in data_list:
#         if data['label'] == label:
#             category = data['image_path'].split('/')[-2]
#             category_list.append(category)
#             path_list.append(data['image_path'])
#             tensors.append(data['cls_token_embedding'])
#     tensors = np.array(tensors)
#     return tensors, category_list, path_list

# def calculate_accuracy(retrieval_category, test_category_list, k_list=[5, 10, 20, 40, 100]):
#     accuracies = {k: 0 for k in k_list}
    
#     for i, test_category in enumerate(test_category_list):
#         for k in k_list:
#             retrieved_categories = retrieval_category[i][:k]
#             correct_retrievals = retrieved_categories.count(test_category)
#             accuracies[k] += correct_retrievals / k

#     # 计算每个 k 的平均准确率
#     for k in accuracies:
#         accuracies[k] /= len(test_category_list)
    
#     return accuracies

# train_path = '/mnt/data_llm/101_train_path.json'
# test_path = '/mnt/data_llm/101_test_path.json'

# # 加载数据
# data_list = load_json('/mnt/data_llm/food101_clip_vit-base_embeddings.json')

# # 提取训练和测试数据
# train_tensors, train_category_list, train_path_list = get_tensors(data_list, 'train')
# test_tensors, test_category_list, test_path_list = get_tensors(data_list, 'test')

# # 保存训练和测试的路径
# save_json(train_path, train_path_list)
# save_json(test_path, test_path_list)

# # 计算相似度排序索引
# sorted_indices = compute_similarity(test_tensors, train_tensors)

# # 生成三个存储结果的列表
# retrieval_category = []
# retrieval_image = []
# retrieval_index = []

# for i, test_path in tqdm(enumerate(test_path_list), total=len(test_path_list)):
#     sorted_idx = sorted_indices[i][:100]  # 取前100个索引
    
#     # 按照排序后的索引获取对应的类别和路径
#     sorted_categories = [train_category_list[idx] for idx in sorted_idx]
#     sorted_paths = [train_path_list[idx] for idx in sorted_idx]
    
#     retrieval_category.append(sorted_categories)  # 修正：添加为列表
#     retrieval_image.append({test_path: sorted_paths})
#     retrieval_index.append({test_path: sorted_idx.tolist()})

# # 计算不同 k 值的准确率
# accuracy_k_list = [5, 10, 20, 40, 100]
# accuracies = calculate_accuracy(retrieval_category, test_category_list, accuracy_k_list)

# # 输出结果
# for k in accuracy_k_list:
#     print(f'Accuracy for top-{k} retrievals: {accuracies[k]:.4f}')

# # 保存到 JSON 文件
# save_json('/mnt/data_llm/172_image_retrieval_category.json', retrieval_category)
# save_json('/mnt/data_llm/172_image_retrieval_image.json', retrieval_image)
# save_json('/mnt/data_llm/172_image_retrieval_index.json', retrieval_index)

# def load_txt(filename):
#     with open(filename, 'r') as f:
#         lines = f.readlines()
#         return [line.strip() for line in lines]

# retrieval_category = load_json('/mnt/data_llm/172_image_retrieval_category_old.json')
# categories = load_txt('/mnt/data_llm/food_images/VireoFood172/SplitAndIngreLabel/FoodList.txt')


# retrieval_category_new = []
# for item in retrieval_category:
#     for key, value in item.items():
#         new_key = key
#         new_value = []
#         for v in value:
#             new_value.append(categories[int(v) - 1])
#     item_new = {new_key: new_value}
#     retrieval_category_new.append(item_new)
# retrieval_category_new_path = '/mnt/data_llm/172_image_retrieval_category.json'
# save_json(retrieval_category_new_path, retrieval_category_new)

            
###############################################################检索增强的LVLMs的questions#############################################################


food101_question_path = '/mnt/data_llm/json_file/101_questions.jsonl'


questions = load_jsonl(food101_question_path)

tamplate = "The categories of the k images most similar to this image are:"
question_tamplate = "Based on the information above, please answer the following questions. What dish is this? Just provide its category."
def convert_q2q(questions, retrieval_category_path, k):
    retrieval_category = load_json(retrieval_category_path)
    new_questions = []
    test_path_idx = load_json('/mnt/data_llm/101_test_path_idx.json')
    #print(test_path_idx)
    for question in tqdm(questions, total=len(questions)):
        new_question = question
        # Assume question contains an image path we need to match
        image_path = question.get('image')
        new_text = tamplate
        idx = test_path_idx[image_path]
        item = retrieval_category[idx]
        image_path = image_path.replace('/media/fast_data/VireoFood172/ready_chinese_food/', '/mnt/data_llm/food_images/VireoFood172/ready_chinese_food/')
        for key, value in item.items():
            if key == image_path:
                for idx, category in enumerate(value[:k]):
                    category = category.replace('_', ' ')
                    if idx < k - 1:
                        new_text += category + ', '
                    if idx == k-1:
                        new_text += "and " + category + '. '
        # # Find the matching category based on image path
        # for item in retrieval_category:
        #     for key, value in item.items():
        #         if key == image_path:
        #             for idx, category in enumerate(value[:k]):
        #                 if idx < k - 1:
        #                     new_text += category + 'and'
        #                 if idx == k-1:
        #                     new_text += category + '.'
        #             break
        new_text += question_tamplate
        new_question['text'] = new_text
        new_questions.append(new_question)
    
    return new_questions

food101_retrieval_category_path = '/mnt/data_llm/101_image_retrieval_category.json'
food101_new_question_path = '/mnt/data_llm/json_file/101_questions_retrieval_1.jsonl'
new_questions = convert_q2q(questions, food101_retrieval_category_path, 1)
save_jsonl(food101_new_question_path, new_questions)

# test_paths = load_json('/mnt/data_llm/172_test_path.json')
# test_path_dict = {}
# for idx, test_path in enumerate(test_paths):
#     test_path = test_path.replace('/mnt/data_llm/food_images/VireoFood172/ready_chinese_food/', '/media/fast_data/VireoFood172/ready_chinese_food/')
#     test_path_dict[test_path] = idx
    
# test_paths_idx = '/mnt/data_llm/172_test_path_idx.json'
# save_json(test_paths_idx, test_path_dict)


# food101_question_path = '/mnt/data_llm/json_file/172_questions.jsonl'


# questions = load_jsonl(food101_question_path)

# new_questions = []
# for question in questions:
#     new_question = question
#     text = "What dish is this? Please answer the question as briefly as possible, Do not provide Paragraph"
#     new_question["text"] = text
#     new_questions.append(new_question)
# food101_question_path_new = '/mnt/data_llm/json_file/172_questions_noft_v1.jsonl'
# save_jsonl(food101_question_path_new, new_questions)