import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image

class VOCDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, grid_size, 
                        num_bboxes, num_classes, transforms=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.grid_size = grid_size
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc(index, 1))
        bboxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(float(x))
                    for x in label.replace('\n', '').split() 
                ]
                bboxes.append([class_label, x, y, width, height])
        img_path = os.path.join(self.img_dir, self.annotations.iloc(index, 0))
        image = Image.open(img_path)
        bboxes = torch.tensor(bboxes)
        
        if self.transforms:
            image, bboxes = self.transforms(image, bboxes)

        label = torch.zeros((self.grid_size, self.grid_size, self.num_classes + 5))

        for bbox in bboxes:
            