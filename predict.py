## 预测测试集结果
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataSet import MultiModalDataSet
from model.MultiModal import  MultiModalClassify

batch_size = 32  # 批次大小
test_dataset = MultiModalDataSet('data/test.csv')  
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 创建测试数据加载器，不打乱顺序

# 创建模型实例，并移动到GPU上（如果有）
model = MultiModalClassify()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.load_state_dict(torch.load('model/net_05.pth'))

# 定义一个测试函数，输入模型、数据加载器和损失函数，返回预测概率
def predict_func(model, loader):
    test_proba = []
    # 设置模型为评估模式，不使用 dropout 层
    model.eval()
    # 遍历测试数据加载器，获取每个批次的数据
    for batch in tqdm(loader):
        # 获取图片、文本和标签，并移动到 GPU 上（如果有）
        feature = batch["feature"].to(device)
        label = batch["label"].to(device)
        # 前向传播，得到模型输出
        output = model(feature)
        # 将模型输出转换为概率，并添加到列表中
        proba = F.softmax(output, dim=-1).detach().cpu().numpy()
        test_proba.extend(proba)
    return  test_proba

test_proba = predict_func(model, test_loader)
# 使用 argmax 函数，指定 axis 参数为 1，得到每一行概率最大的索引
label = np.argmax(test_proba, axis=1)
# 定义一个 id_to_label 字典，将数字标签映射为文本标签
id_to_label = {0:"negative", 1:"neutral", 2:"positive"}
# 使用列表推导式，根据 id_to_label 字典将每个数字标签映射为对应的文本标签
text_label = [id_to_label[i] for i in label]

guid_list = pd.read_csv('data/origin_data/test_without_label.txt')
with open('predict_result.txt','w',encoding='utf-8') as file:
    file.write("guid,tag\n")
    for i in range(511):
        file.write(str(guid_list['guid'][i]))
        file.write(",")
        file.write(text_label[i])
        file.write("\n")

