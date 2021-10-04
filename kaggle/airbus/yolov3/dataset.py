import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2



class AirbusDataset(Dataset):
    def __init__(self, image_dir, csv_dir):
        self.annotations = pd.read_csv(csv_dir)
        self.image_dir = image_dir


    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.annotations['ImageId'][index])
        image = cv2.imread(image_path)

        label = self.annotations.iloc(index, 1).split()
        mask = np.zeros(image.shape()[0] * image.shape()[1], dtype=np.uint8)
        for i in range(0, len(label), 2):
            mask[int(label[i]) : int(label[i]) + int(label[i + 1])] = 1
        mask = mask.reshape(image.shape()[0 : 2]).T

        x, y, w, h = cv2.boundingRect(mask)

        # bbox = 


if __name__ == '__main__':
    annotations = pd.read_csv('/home/neik/data/airbus-ship-detection/train_ship_segmentations_v2.csv')
    image_dir = '/home/neik/data/airbus-ship-detection/train_v2/'
    image_path = os.path.join(image_dir, annotations['ImageId'][0])
    print(image_path)