from torchvision import models
import torch
import cv2
from albumentations import Resize, Compose
from albumentations.pytorch.transforms import ToTensor
from albumentations.augmentations.transforms import Normalize
import onnx
import time

def preprocess_img(img_path):
    transforms = Compose([
        Resize(224, 224, interpolation=cv2.INTER_NEAREST),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor(),
    ])

    input_img = cv2.imread(img_path)
    input_data = transforms(image=input_img)["image"]

    batch_data = torch.unsqueeze(input_data, dim=0)
    return batch_data


def postprocess(output_data):
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100
    _, indices = torch.sort(output_data, descending=True)
    i = 0
    while confidences[indices[0][i]] > 0.5:
        class_idx = indices[0][i]
        # print(
        #     'class: ', classes[class_idx],
        #     ', confidence: ', confidences[class_idx].item(),
        #     '%, index: ', class_idx.item(),
        # )
        i += 1
if __name__ == '__main__':
    model = models.resnet50(pretrained=True)
    model.eval()
    model.cuda()
    start_time = time.time()
    for i in range(10000):
        input = preprocess_img('turkish_coffee.webp').cuda()
        output = model(input)
        postprocess(output)
    # print('[DEBUG MESSAGE] {}:{}'.format('', postprocess(output)))
    end_time = time.time()
    print('Time taken: {} seconds'.format(end_time - start_time))

    # Convert to ONNX format
    # ONNX_FILE_PATH = 'resnet50.onnx'

    # torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=['input'], 
    #                     output_names=['output'], export_params=True)
    # onnx_model = onnx.load(ONNX_FILE_PATH)
    # onnx.checker.check_model(onnx_model)
