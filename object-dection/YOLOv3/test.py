import torch
import cv2
import numpy as np

import config
from model import YOLOv3
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)

torch.backends.cudnn.benchmark = True

def test(model, image):
    out_scale_0, out_scale_1, out_scale_2 = model(image)
    out_wrt_img_0 = cells_to_bboxes(out_scale_0, config.ANCHORS[0], self.config.S[0])
    out_wrt_img_1 = cells_to_bboxes(out_scale_1, config.ANCHORS[1], self.config.S[1])
    out_wrt_img_2 = cells_to_bboxes(out_scale_2, config.ANCHORS[2], self.config.S[2])
    print(out_wrt_img_0)


def main():
    model = YOLOv3(num_classes=20).to(config.DEVICE)
    checkpoint = torch.load('./my_checkpoint.pth.tar', map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    # model.eval()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv",
        test_csv_path=config.DATASET + "/test.csv"
    )

    pred_boxes, true_boxes = get_evaluation_bboxes(
        test_loader,
        model,
        iou_threshold=config.NMS_IOU_THRESH,
        anchors=config.ANCHORS,
        threshold=config.CONF_THRESHOLD,
    )

if __name__ == '__main__':
    main()