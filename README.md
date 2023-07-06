### 实验五：多模态情感分析

data 文件夹 ：
- origin_data 文件夹 ：
  - data : 成对的图像（jpg 文件）和文本（txt 文件）数据
  - test_without_label.txt : 测试集数据
  - train.txt : 训练集、验证集数据
- __init__.py : 用于将文件夹变为一个 Python 模块
- dataSet.py ：原始图像、文本数据特征提取、融合方法
- test.csv ：测试集相关信息
- train.csv ：训练集相关信息
- valid.csv ：验证集相关信息

model 文件夹 ：
- pretrain 文件夹 ：bert 预训练模型
  - bert-base-uncased : bert 模型变体
    - bert_config.json : 原版模型配置
    - config.json : 模型配置文件 
    - pytorch_model.bin : bert 模型权重（该文件太大不支持上传到 GitHub，如果需要[点击此处](https://blog.csdn.net/fghjbjhgb/article/details/107850137)自行上网下载或联系本人）
    - tokenizer.json : bert 词汇表
- __init__.py : 用于将文件夹变为一个 Python 模块
- MultiModal.py : 多模态融合模型
- net_01.pth : 第一轮训练模型参数
- net_02.pth : 第二轮训练模型参数
- net_03.pth : 第三轮训练模型参数
- net_04.pth : 第四轮训练模型参数
- net_05.pth : 第五轮训练模型参数
  
data_deal.py : 数据相关信息读取处理，执行可获得用于后续模型训练的数据信息

external.py : 消融代码，运行可得到只输入文本或图像数据，多模态融合模型在验证集的表现情况

predict_result.txt ：预测结果文件

predict.py : 测试代码，运行可得到预测结果文件

requirements.txt ：执行代码所依赖的主要库

train.py ：训练、验证代码，运行可得到新的模型参数文件及训练、验证准确率随轮次变化曲线

实验报告.pdf ：实验报告

关于代码功能都在上述指出，当前目录下的四个 py 文件即所有待执行代码，执行一遍完整流程的先后顺序为 data_deal.py -> train.py -> predict.py -> external.py。

联系方式（个人邮箱）：10205501450@stu.encu.edu.cn