import torchvision.models as models
import torch.nn as nn

def resnet(out_feature=2, feature=18, pretrained=True): # 定义模型
    assert feature in [18,34,50,101, 152], 'ResNet don\'t consist of {}'.format(feature) # 确保模型存在
    if feature == 18:
        model_fit = models.resnet18(pretrained=pretrained) # 调用库中已有模型，并开启预训练，会下载一个预训练文件
    elif feature == 34:
        model_fit = models.resnet34(pretrained=pretrained)
    elif feature == 50:
        model_fit = models.resnet50(pretrained=pretrained)
    elif feature == 101:
        model_fit = models.resnet101(pretrained=pretrained)
    elif feature == 152:
        model_fit = models.resnet152(pretrained=pretrained)
    num_ftrs = model_fit.fc.in_features # 获取最后一个全连层输入特征数
    model_fit.fc = nn.Linear(num_ftrs, out_feature) # 根据类别数，重定义最后一个全列层
    return model_fit
