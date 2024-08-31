from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm

# 根目录路径
#root_dir = '/mnt/data_llm/food_images/VireoFood172/ready_chinese_food'
# root_dir = '/media/fast_data/food_recognition_dataset/food-101/images/'
root_dir = '/mnt/data_llm/food_images/food200/images/'
# 训练数据和测试数据文件路径
# train_file = '/mnt/data_llm/food_images/VireoFood172/SplitAndIngreLabel/TR.txt'
# test_file = '/mnt/data_llm/food_images/VireoFood172/SplitAndIngreLabel/TE.txt'
# train_file = '/media/fast_data/food_recognition_dataset/food-101/meta/train.txt'
# test_file = '/media/fast_data/food_recognition_dataset/food-101/meta/test.txt'
train_file = '/mnt/data_llm/food_images/food200/metadata/train_finetune_v2.txt'
test_file = '/mnt/data_llm/food_images/food200/metadata/test_finetune_v2.txt'

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


# 初始化处理器
processor = AutoImageProcessor.from_pretrained('/media/fast_data/model/dinov2-large')

# 创建训练和测试数据集
train_dataset = FoodDataset(train_file, root_dir, "train", processor)
test_dataset = FoodDataset(test_file, root_dir, "test", processor)

# 合并数据集
full_dataset = train_dataset + test_dataset

# 定义DataLoader
batch_size = 16
dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=64)

# 初始化模型并放置在GPU上
model = AutoModel.from_pretrained('/media/fast_data/model/dinov2-large')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 用于存放数据的列表
data_list = []

# 使用tqdm添加进度条
for batch in tqdm(dataloader, desc="Processing images"):
    # 获取图像路径和标签
    try:
        batch_labels = batch.pop('label')
        batch_paths = batch.pop('image_path')
        batch_pixel_values = batch['pixel_values'].to(device)
        batch_pixel_values = batch_pixel_values.squeeze(1)  # 移除第二维度(1)
        # 使用模型获取输出
        outputs = model(pixel_values=batch_pixel_values)
        
        # 获取 cls_token_embedding 和 average_embedding
        last_hidden_states = outputs.last_hidden_state
        cls_token_embeddings = last_hidden_states[:, 0, :].detach().cpu().numpy().tolist()  # 转回CPU
        average_embeddings = torch.mean(last_hidden_states, dim=1).detach().cpu().numpy().tolist()  # 转回CPU
        
        # 为当前批次创建字典并加入到列表中
        for i in range(len(batch_paths)):
            data = {
                'image_path': batch_paths[i],
                'label': batch_labels[i],  # 添加训练/测试标签
                'cls_token_embedding': cls_token_embeddings[i],
                'average_embedding': average_embeddings[i]
            }
            data_list.append(data)
    except:
        print(1111)

# 将数据保存到 JSON 文件
with open('/mnt/data_llm/food200_embeddings.json', 'w') as json_file:
    json.dump(data_list, json_file, indent=4)

print("数据已成功保存到 JSON 文件中")
