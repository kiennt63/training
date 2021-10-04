import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

def get_iou(bbox1, bbox2):
    assert bbox1['x1'] < bbox1['x2']
    assert bbox1['y1'] < bbox1['y2']
    assert bbox2['x1'] < bbox2['x2']
    assert bbox2['y1'] < bbox2['y2']
    s1 = (bbox1['x2'] - bbox1['x1']) * (bbox1[y2] - bbox1[y1])
    s2 = (bbox2['x2'] - bbox2['x1']) * (bbox2[y2] - bbox2[y1])

    x_left = max(bbox1['x1'], bbox2['x1'])
    x_right = min(bbox1['x2'], bbox2['x2'])
    y_top = max(bbox1['y1'], bbox2['y1'])
    y_bottom = min(bbox1['y2'], bbox2['y2'])

    if x_left > x_right or y_top > y_bottom:
        return 0.0
    
    i = (x_right - x_left) * (y_bottom - y_top)
    u = s1 + s2 - i

    iou = i / u
    assert iou >= 0
    assert iou <= 1

    return iou

class AirbusDataset(Dataset):
    def __init__(self, image_dir, csv_dir):
        self.annotations = pd.read_csv(csv_dir)
        self.image_dir = image_dir


    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.annotations.iloc(index, 0))
        image = cv2.imread(image_path)

        label = self.annotations.iloc(index, 1).split()
        mask = np.zeros(image.shape()[0] * image.shape()[1], dtype=np.uint8)
        for i in range(0, len(label), 2):
            mask[int(label[i]) : int(label[i]) + int(label[i + 1])] = 1
        mask = mask.reshape(image.shape()[0 : 2]).T

        x, y, w, h = cv2.boundingRect(mask)

        bbox = 




if __name__ == '__main__':
    data_csv = pd.read_csv(os.environ['DATA_PATH'] + 'airbus-ship-detection/train_ship_segmentations_v2.csv')
    print(data_csv)



