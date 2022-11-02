import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import torch
def build_dataset(data_type, args): # 数据读取
    """args: args.dataset"""
    simple_transform = transforms.Compose([transforms.Scale((224,224)), transforms.ToTensor(),  
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])    # transforms.Compose做图片处理和增强
    assert data_type in ["train", "valid"]
    if data_type is "train":
        data = datasets.ImageFolder(os.path.join(args.dataset, 'train'), simple_transform) # dogsandcats/train/
    else:
        data = datasets.ImageFolder(os.path.join(args.dataset, 'valid'), simple_transform)
    return data

# import argparse

# parser = argparse.ArgumentParser("tests")
# parser.add_argument("--dataset", type=str, default="dogsandcats", help="sim")
# args = parser.parse_args()
# data = build_dataset('train', args=args)
# for i in torch.utils.data.DataLoader(data):
#     print(i[0].shape)