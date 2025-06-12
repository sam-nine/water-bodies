import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)

        smooth = 1e-5
        inputs = torch.sigmoid(inputs)  # Needed for dice
        targets = targets.type_as(inputs)

        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        dice_score = (2. * intersection + smooth) / (
            inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + smooth
        )
        dice_loss = 1 - dice_score.mean()

        return bce_loss + dice_loss
