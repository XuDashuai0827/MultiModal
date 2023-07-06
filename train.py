import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataSet import MultiModalDataSet
from model.MultiModal import MultiModalClassify
import matplotlib.pyplot as plt

# 定义超参数
batch_size = 32  # 批次大小
num_epochs = 5  # 训练轮数
learning_rate = 0.01  # 学习率

# 创建数据集和数据加载器
train_dataset = MultiModalDataSet('data/train.csv') 
valid_dataset = MultiModalDataSet('data/valid.csv') 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 创建训练数据加载器，打乱顺序
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)  # 创建验证数据加载器，不打乱顺序

# 创建模型实例，并移动到 GPU 上（如果有）
model = MultiModalClassify()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数，适用于多分类问题
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 使用 Adam 优化器，自适应调整学习率


# 定义一个训练函数，输入模型、数据加载器、损失函数和优化器，返回训练损失、准确率和预测概率
def train(model, loader, criterion, optimizer):
    # 初始化训练损失和准确率
    train_loss = 0.0
    train_acc = 0.0
    # 定义一个空列表，用于存储预测概率
    train_proba = []
    model.train()
    # 遍历训练数据加载器，获取每个批次的数据
    for batch in tqdm(loader):
        # 获取图片、文本和标签，并移动到 GPU 上（如果有）
        feature = batch["feature"].to(device)
        label = batch["label"].to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播，得到模型输出
        output = model(feature)
        # 计算损失
        loss = criterion(output, label)
        # 反向传播，计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        # 累加训练损失和准确率
        train_loss += loss.item()
        train_acc += (output.argmax(dim=-1) == label).sum().item()
        # 将模型输出转换为概率，并添加到列表中
        proba = F.softmax(output, dim=-1).detach().cpu().numpy()
        train_proba.extend(proba)
    # 计算训练损失和准确率的平均值
    train_loss /= len(loader)
    train_acc /= len(loader.dataset)
    return train_loss, train_acc, train_proba


# 定义一个测试函数，输入模型、数据加载器和损失函数，返回测试损失、准确率和预测概率
def valid(model, loader, criterion):
    # 初始化测试损失和准确率
    valid_loss = 0.0
    valid_acc = 0.0
    # 定义一个空列表，用于存储预测概率
    valid_proba = []
    # 设置模型为评估模式，不使用 dropout 层
    model.eval()
    # 遍历测试数据加载器，获取每个批次的数据
    for batch in tqdm(loader):
        # 获取图片、文本和标签，并移动到 GPU 上（如果有）
        feature = batch["feature"].to(device)
        label = batch["label"].to(device)
        # 前向传播，得到模型输出
        output = model(feature)
        # 计算损失
        loss = criterion(output, label)
        # 累加测试损失和准确率
        valid_loss += loss.item()
        valid_acc += (output.argmax(dim=-1) == label).sum().item()
        # 将模型输出转换为概率，并添加到列表中
        proba = F.softmax(output, dim=-1).detach().cpu().numpy()
        valid_proba.extend(proba)
    print('Saving model......')
    torch.save(model.state_dict(), 'model/net_%02d.pth' % (epoch + 1))

    # 计算测试损失和准确率的平均值
    valid_loss /= len(loader)
    valid_acc /= len(loader.dataset)
    return valid_loss, valid_acc, valid_proba


train_acc_list = []
valid_acc_list = []

# 循环训练指定轮数
print("start MultiModal training")
print("============================")
for epoch in range(num_epochs):
    # 调用训练函数，得到训练损失、准确率和预测概率
    print(f'Epoch [{epoch + 1}/{num_epochs}] training start:')
    train_loss, train_acc, train_proba = train(model, train_loader, criterion, optimizer)
    # 调用测试函数，得到测试损失、准确率和预测概率
    print(f'Epoch [{epoch + 1}/{num_epochs}] validating start:')
    valid_loss, valid_acc, valid_proba = valid(model, valid_loader, criterion)
    # 打印每个 epoch 的损失和准确率
    print(f'Epoch [{epoch + 1}/{num_epochs}], train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')
    print(f'Epoch [{epoch + 1}/{num_epochs}], valid loss: {valid_loss:.4f}, valid acc: {valid_acc:.4f}')

    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

# 绘制每个 epoch 训练集和测试集的准确率值变化曲线，设置颜色、标签和线宽
plt.title("train & valid accuracy curve")
plt.plot(range(1,num_epochs+1), train_acc_list, color='blue', label='train', linewidth=2)
plt.plot(range(1,num_epochs+1), valid_acc_list, color='red', label='valid', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
