import json

file_path = '/media/fast_data/datacomp_sample/datacomp_1b.json'
with open(file_path, 'r') as file:
    data = json.load(file)

data_new = []
for item in data:
    data = item
    data['image'] = item['image'].replace('datacomp_sample/datacomp_sample', 'datacomp_sample')
    data_new.append(data)

file_path = '/media/fast_data/datacomp_sample/datacomp_1b_new.json'
with open(file_path, 'w') as file:
    json.dump(data_new, file, indent=4)