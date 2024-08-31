# import json


# food200_train_file = '/media/fast_data/food_recognition_dataset/food200/metadata/train_finetune_v2.txt'

# with open (food200_train_file, 'r') as f:
#     lines = f.readlines()




# idx = 0
# root_path = '/media/fast_data/food_recognition_dataset/food200/images'
# for line in lines:
#     line = line.strip()
#     category, file_name_and_index = line.split('/')
#     file_name, index = file_name_and_index.split()
#     iamge_path = root_path + '/' + category + '/' + file_name
#     prompt = "<image>\n" + 
#     conversation = {
#         "id": str(idx),
#         "image": iamge_path,
#         "conversations": [
#             {
#                 "from": "human",
#                 "value": prompt,
#             },
#             {
#                 "from": "gpt",
#                 "value": category
#             }
#         ]]
#     }
#     idx += 1