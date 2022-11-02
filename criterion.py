import torch.nn as nn
import torch

CEL = nn.CrossEntropyLoss()

def build_criterion(outputs, targets):
    loss = CEL(outputs, targets)
    _, preds = torch.max(outputs.data, 1)
    batch_corrects = (torch.sum(preds == targets.data)).data.cpu().numpy()
    return loss, batch_corrects
    