import torch
import torchvision
import transformers
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# 图像、文本特征提取并融合
class MultiModalDataSet(Dataset):
    def __init__(self, csv_file, img_transform=None, text_transform=None, need_text=True, need_image=True):
        self.data = pd.read_csv(csv_file)
        self.need_text = need_text
        self.need_image = need_image
        self.img_transform = img_transform or torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Grayscale(num_output_channels=3),  # 将图片转换为 3 通道的灰度图像
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 使用 AutoTokenizer 和 AutoModel 自动加载预训练的 BERT 模型
        self.text_transform = text_transform or transformers.AutoTokenizer.from_pretrained("model\\pretrain\\bert-base-uncased", local_files_only=True)
        self.text_model = transformers.AutoModel.from_pretrained("model\\pretrain\\bert-base-uncased", local_files_only=True)

        # 加载预训练的 resNet18 模型，并将最后一层替换为一个线性层，用于将图片向量变换为 1000 维
        self.image_model = torchvision.models.resnet18(pretrained=True)
        self.image_model.eval()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        if image_path and self.need_image:
            image = Image.open(image_path)
            image = self.img_transform(image)
            image = self.image_model(image.unsqueeze(0)) 
            image = image[0] # 去掉第一个维度，得到一维向量作为图片特征
        else:
            image = torch.zeros(1000)

        text = self.data.iloc[idx, 1]
        if text and self.need_text:
            # 对文本进行编码、填充和截断，并返回张量
            text = self.text_transform.encode_plus(text,return_tensors="pt",padding="max_length",truncation=True,max_length=256)
            # 对文本进行编码，得到输出特征
            text_output = self.text_model(**text)
            text_feature = text_output[0][:,0,:] # 取第一个隐藏层作为文本特征
            text_feature= text_feature.squeeze(0)
        else:
            text_feature = torch.zeros(768)

        label = int(self.data.iloc[idx, 2])
        feature = torch.cat([text_feature, image], dim=0)

        sample = {"feature":feature, "label": label}
        return sample
