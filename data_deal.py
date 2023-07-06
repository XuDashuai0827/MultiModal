import pandas as pd
import os
from bs4 import UnicodeDammit
from sklearn.model_selection import train_test_split

## 处理数据
def train_split(file,split=True):
    train = pd.read_csv(file)
    img_path = []
    text_content = []
    label = []
    label_to_id ={"negative":0,"neutral":1,"positive":2}

    for index , row in train.iterrows(): # 对每行（训练或测试）文本数据循环
        id = str(int(row ["guid"]))
        # 找到 guid 对应的图像和文本数据所在路径
        img = os.path.join("data/origin_data/data",id+".jpg")
        text = os.path.join("data/origin_data/data",id+".txt")
        img_path.append(img)
        try :
            label.append(label_to_id[row["tag"]]) # 将每行对应的标签转换为 0、1 或 2，方便分类
        except:
            label.append(-1) # 遇到标签为 null 在 label_to_id 找不到对应转换规则的数据（测试集数据），则先把标签记为 -1
        with open(text,"rb") as f:
            dammit = UnicodeDammit(f.read())
            text_content.append(dammit.unicode_markup)
    train["image"]=img_path
    train["text"] =text_content
    train["label"] =label

    if split:
        train_data, test_data = train_test_split(train, train_size=0.8, random_state=50, stratify=train['label'])
        # 保存训练集和验证集为 csv 文件
        train_data.to_csv("data/train.csv",columns=["image","text","label"],sep=",",index=False)
        test_data.to_csv("data/valid.csv",columns=["image","text","label"],sep=",",index=False)
    else:
        train.to_csv("data/test.csv", columns=["image", "text", "label"], sep=",", index=False)

train_split("data/origin_data/train.txt",split=True)
train_split("data/origin_data/test_without_label.txt",split=False)
