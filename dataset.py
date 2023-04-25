from  torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
from PIL import Image
class Ocean_data(Dataset):
    def __init__(self, data_dir, label_dir, flag):
        super().__init__()
        self.name_list = os.listdir(data_dir)  # 获得子目录下的图片的名称
        self.label = np.loadtxt(label_dir,delimiter=',')
        self.imgpath = data_dir
        self.flag = flag
        self.transform = transforms.Compose(
            [
                # transforms.Resize(size = (512,512)),#尺寸规范
                # transforms.RandomResizedCrop((512,512)),
                # transforms.RandomCrop((512,512), padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # transforms.RandomRotation(45),
                # transforms.ColorJitter(contrast=0.5),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),   #转化为tensor
                # transforms.Normalize((0.5), (0.5)),
                
            ])# Transforms只适用于PIL 中的Image打开的图像
    def __getitem__(self, index):
        name = self.name_list[index]  # 获得当前图片的名称
        num = int(name[:-4])
        path = os.path.join(self.imgpath,name)
        image = Image.open(path)
        # image = np.expand_dims(image,axis=0)
        # image = torch.FloatTensor(image).permute(2,0,1)
        image = self.transform(image)
        if self.flag == 'human':
            if self.label[num][1] == 1:
                label = 1
            else:
                label = 0
        elif self.flag == 'ship':
            if self.label[num][2] == 1:
                label = 1
            else:
                label = 0
        else:
            if self.label[num][2] == 1 or self.label[num][1] == 1:
                label = 1
            else:
                label = 0
        # label =  np.reshape(label,(1,))
        label = torch.as_tensor(label, dtype=torch.int64)
        # label = torch.FloatTensor(label)
        return image,label

    def __len__(self):
        return len(self.name_list)