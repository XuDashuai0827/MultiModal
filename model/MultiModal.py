from torch import nn
import torch.nn.functional as F

# 多模态融合模型架构
class MultiModalClassify(nn.Module):
    def __init__(self):
        super(MultiModalClassify, self).__init__()
        self.fc1 = nn.Linear(1768,512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 3)

    def forward(self,x):
        # 前向传播函数，输入图片和文本，输出分类结果
        # 提取融合特征的特征，得到一个 512 维的向量
        hidden = self.fc1(x)
        # 通过激活函数，增加非线性
        hidden = F.relu(hidden)
        # 通过 dropout 层，防止过拟合
        hidden = self.dropout(hidden)
        # 通过全连接层，得到一个 3 维的向量
        output = self.fc2(hidden)
        return output
