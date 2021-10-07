import torch
import torch.nn as nn
from utils import compute_iou

class YoloLoss(nn.Module):
    def __init__(self, grid_size, num_boxes, num_classes):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.s = grid_size
        self.b = num_boxes
        self.c = num_classes
        self.lambda_coord = 5
        self.lambda_no_obj = 0.5

    def forward(self, prediction, target):
        prediction = prediction.reshape(-1, self.s, self.s, self.c + self.b * 5) # N * S * S * 30

        iou_bbox1 = compute_iou(prediction[..., 21:25], target[..., 21:25]) # N * S * S * 4
        iou_bbox2 = compute_iou(prediction[..., 26:30], target[..., 21:25]) # N * S * S * 4
        ious = torch.cat(iou_bbox1.unsqueeze(0), iou_bbox2.unsqueeze(0), dim=0) # 2 * N * S * S * 4
        iou_max, best_box = torch.max(ious, dim=0) # find index of the best bbox
        exist_box = target[..., 20:21]

        # ---------------- Coordinate loss ----------------
        box_prediction = exist_box * (best_box * prediction[..., 26:30] + (1 - best_box) * prediction[..., 21:25])
        box_target = exist_box * target[..., 21:25]

        box_prediction[..., 2:4] = torch.sign(box_prediction[..., 2:4]) * torch.sqrt(torch.abs(box_prediction[..., 2:4] + 1e-6))

        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])

        coord_loss = self.mse(
            torch.flatten(box_prediction, end_dim=-2),
            torch.flatten(box_target, end_dim=-2)
        )

        # ---------------- Object loss ----------------
        # Have object
        pred_obj_score = best_box * prediction[..., 25:26] + (1 - best_box) * prediction[..., 20:21]
        target_obj_score = target[..., 20:21]

        object_loss = self.mse(
            torch.flatten(pred_obj_score),
            torch.flatten(target_obj_score)
        )
        # No object
        no_object_loss = self.mse(
            torch.flatten((1 - exist_box) * prediction[..., 20:21], start_dim=1),
            torch.flatten((1 - exist_box) * target_obj_score)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exist_box) * prediction[..., 25:26], start_dim=1),
            torch.flatten((1 - exist_box) * target[..., 20:21], start_dim=1)
        )

        # ---------------- Class loss --------------------
        class_loss = self.mse(
            torch.flatten(exist_box * prediction[..., :20], end_dim=-2),
            torch.flatten(exist_box * target[..., :20], end_dim=-2)
        )

        # ---------------- Total loss --------------------
        loss = (
            self.lambda_coord * coord_loss +
            object_loss + 
            self.lambda_no_obj * no_object_loss +
            class_loss
        )
        
