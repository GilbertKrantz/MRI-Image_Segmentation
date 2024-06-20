import torch

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        
    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        smooth = 1.0
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        intersection = (y_pred * y_true).sum()
        
        dic_loss = 1 - (2.0 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
        return dic_loss