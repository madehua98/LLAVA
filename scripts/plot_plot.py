import matplotlib.pyplot as plt

# 输入数据
models = ["ViT B/16", "FoodViT B/16", "FoodCLIP-B-6-NF", "FoodCLIP-B-6", "FoodCLIP-B"]
performance_food101 = [93.20, 94.09, 95.07, 95.26, 96.26]
performance_food172 = [92.17, 92.28, 92.67, 93.36, 94.18]
performance_food200 = [71.80, 73.80, 74.68, 75.62, 77.05]

# 计算每个模型在三个数据集上的平均性能
average_performances = []
for i in range(len(models)):
    average_performance = (performance_food101[i] + performance_food172[i] + performance_food200[i]) / 3
    average_performances.append(average_performance)

# 绘制折线图
plt.figure(figsize=(7, 5))  # 调整图像大小以减少标签间距
plt.plot(models, average_performances, marker='s', linestyle='--', color='#FF6347')

# 添加每个点的数值
for i in range(len(models)):
    plt.text(models[i], average_performances[i]+0.02, f'{average_performances[i]:.2f}', 
             ha='center', va='bottom', fontsize=10, color='black')

# 设置纵轴标签并添加注释
plt.ylabel('Average Top-1 ACC (%) on three datasets', fontsize=12, labelpad=15)

plt.grid(True)

# 调整标签旋转角度并居中对齐
plt.xticks(rotation=30, ha='center')

plt.tight_layout()  # 调整整体空白

# 保存图像到文件
plt.savefig('/home/data_llm/madehua/LLaVA/scripts/average_performance_plot.png', dpi=300)

# 显示图像
plt.show()
