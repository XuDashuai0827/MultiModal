import torch
from torch.utils.data import DataLoader
from data.dataSet import MultiModalDataSet
from model.MultiModal import  MultiModalClassify
import sklearn.metrics
from predict import predict_func

## 消融实验结果(验证集)
batch_size = 32  # 批次大小
num_epochs = 5  # 训练轮数
learning_rate = 0.01  # 学习率

# 创建模型实例，并移动到 GPU 上（如果有）
model = MultiModalClassify()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.load_state_dict(torch.load('model/net_05.pth'))
# 创建数据集和数据加载器
valid_dataset_without_image = MultiModalDataSet('data/valid.csv', need_image=False)  
valid_dataset_without_text = MultiModalDataSet('data/valid.csv', need_text=False)  
valid_loader_1 = DataLoader(valid_dataset_without_image, batch_size=batch_size, shuffle=False)  
valid_loader_2 = DataLoader(valid_dataset_without_text, batch_size=batch_size, shuffle=False)  

print("text only")
valid_proba_1 = predict_func(model, valid_loader_1)
print("image only")
valid_proba_2 = predict_func(model, valid_loader_2)

# 使用 argmax 函数，指定 axis 参数为 1，得到每一行概率最大的索引
valid_y = valid_dataset_without_image.data.iloc[:, 2].to_numpy()
valid_auc_1 = sklearn.metrics.roc_auc_score(valid_y, valid_proba_1, average='macro', multi_class='ovo')
valid_auc_2 = sklearn.metrics.roc_auc_score(valid_y, valid_proba_2, average='macro', multi_class='ovo')

print("with out image auc is " + str(valid_auc_1))
print("with out text auc is " + str(valid_auc_2))





