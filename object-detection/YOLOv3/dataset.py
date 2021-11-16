import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import pandas as pd

from PIL import Image, ImageFile
import cv2

from utils import (
        cells_to_bboxes,
        iou_width_height,
        non_max_suppression as nms,
        plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YoloDataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir, label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.C = C
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_threshold = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        # Use np.roll to convert anno from class, x, y, w, h -> x, y, w, h, class for Albumentations
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        img = np.array(cv2.imread(img_path))

        if self.transform:
            augmentations = self.transform(image=img, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S] # 6 = p_object, x, y, w, h, class

        # Loop thru each object in the image and find the best anchor for it, then the corresponding predicted bbox
        # will be assigned for predicting the object (incur the classification and coordinate loss)
        # If the anchor is already taken by another object -> not sure to be optimal but really rare
        # Optimal: The object that took that anchor is the object that the anchor matches best
        # Not sure this is the implementation in the original code

        # Here for the objectness score, we're not trying to calculate the target values, we're just trying to
        # find the best box for each object and the boxes that will be ignore. The target value for objectness
        # score would be:
        # - IoU value for best box
        # - 0 for all other box with IoU < threshold
        for box in bboxes:
            # Compute IoU between the object size and each of the anchors
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)

            # Sort anchors from best to worst
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, w, h, class_label = box 
            
            # Do we have an anchor on each scale
            has_anchor = [False, False, False]

            # Process the anchors from best to worst match with current object
            for anchor_idx in anchor_indices: # go from the one with the highest iou
                # what scale it belongs to
                scale_idx = anchor_idx // self.num_anchors_per_scale # scale 0, 1, 2
                anchor_idx_on_scale = anchor_idx % self.num_anchors_per_scale # anchor 0, 1, 2 of scale = scale_idx
                current_scale_grid_size = self.S[scale_idx]

                # Get the cell index where the object centered
                cell_idx_x, cell_idx_y = int(current_scale_grid_size * x), int(current_scale_grid_size * y)

                # If the anchor is already taken by another object
                anchor_taken = targets[scale_idx][anchor_idx_on_scale, cell_idx_y, cell_idx_x, 0]

                # If the anchor is not taken by another object and the current scale idx doesn't have a better anchor
                # Then the the corresponding box is the best box for this current object
                # And the box incur classification and coordinate loss
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_idx_on_scale, cell_idx_y, cell_idx_x, 0] = 1
                    x_cell, y_cell = x * current_scale_grid_size - cell_idx_x, y * current_scale_grid_size - cell_idx_y
                    w_cell, h_cell = w * current_scale_grid_size, h * current_scale_grid_size
                    targets[scale_idx][anchor_idx_on_scale, cell_idx_y, cell_idx_x, 1:5] = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                    targets[scale_idx][anchor_idx_on_scale, cell_idx_y, cell_idx_x, 5] = int(class_label)

                    has_anchor[scale_idx] = True

                # Else if the scale idx already have a better anchor and the anchor is not taken by another object
                # And the IoU is greater than a threshold -> ignore the prediction (incur no loss)
                # Set the GT to -1 for further processing
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_threshold:
                    targets[scale_idx][anchor_idx_on_scale, cell_idx_y, cell_idx_x, 0] = -1

                # Other bboxes corresponding to non-best anchor and IoU < threshold will have objectness GT = 0
            
        return image, tuple(targets)





