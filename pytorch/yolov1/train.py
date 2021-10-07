import torch
import torchvision.transforms as transform
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    compute_iou,
    non_max_suppression,
    mean_average_precision,
    cell_to_image,


)
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

from tqdm import tqdm



# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
WEIGHT_DECAY = 0
NUM_EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
IMG_DIR = '/home/neik/data/VOC/images'
LABEL_DIR = '/home/neik/data/VOC/labels'

def train(train_loader, model, optimizer, loss):
    mean_loss = []
    loop = tqdm(train_loader, leave=True)
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        loss = loss(pred, y)
        mean_loss.append(loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print('Mean loss was {}'.format(sum(mean_loss) / len(mean_loss)))

def main():
    model = Yolov1(grid_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    loss = YoloLoss(grid_size=7, num_boxes=2, num_classes=20).to(DEVICE)

    train_dataset = VOCDataset(
        '/home/neik/data/VOC/8examples.csv',
        # transforms=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss)
        
