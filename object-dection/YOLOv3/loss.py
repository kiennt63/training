import torch
import torch.nn as nn

from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss() # for box pred
        self.bce = nn.BCEWithLogitsLoss() # apply sigmoid as well
        self.entropy = nn.CrossEntropyLoss()

        self.sigmoid = nn.Sigmoid()

        # Constants
        self.lambda_class = 1
        self.lambda_no_obj = 10
        self.lambda_obj = 1
        self.lambda_coord = 10

    # Compute the loss for each of 3 scales
    # anchors is the anchors for all scales - shape: (3 x 2) - num_scales x (w & h)
    def forward(self, preds, target, anchors):
        # Get only target of best anchors and anchors with IoU < threshold 
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # No object loss
        no_obj_loss = self.bce(
            (preds[..., 0:1][noobj]),
            (target[..., 0:1][noobj])
        )

        # Object loss
        anchors = anchors.reshape(1, 3, 1, 1, 2) # reshape to match the dimension of predicted values

        # p_w * exp(t_w)
        # p_w: anchor width
        # t_w: network output for w value
        box_preds = torch.cat([self.sigmoid(preds[..., 1:3]), torch.exp(preds[..., 3:5]) * anchors], dim=-1)

        # Calculate the target IoU value | .detach() to make sure gradient not go thru box_preds
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()

        object_loss = self.bce(
            (preds[..., 0:1][obj]),
            (ious * target[..., 0:1][obj])
        )

        # Coordinate loss
        # In stead of taking the loss of p_w * exp(t_w) to the target, modify the target by target = log(target / p_w)
        # This -> better gradient flow
        preds[..., 1:3] = self.sigmoid(preds[..., 1:3]) # x, y to be [0, 1]
        target[..., 3:5] = torch.log(1e-16 + target[..., 3:5] / anchors) # 1e-16 to make sure value inside log > 0

        coord_loss = self.mse(preds[..., 1:5][obj], target[..., 1:5][obj])

        # Classification loss
        class_loss = self.entropy(
            (preds[..., 5:][obj]),
            (target[..., 5][obj].long())
        )

        return (
            self.lambda_coord * coord_loss
            + self.lambda_obj * object_loss
            + self.lambda_no_obj * no_obj_loss
            + self.lambda_class * class_loss
        )

