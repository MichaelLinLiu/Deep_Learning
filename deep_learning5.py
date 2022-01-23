import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
import random
import torch


def get_coloured_mask(mask):
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def predictor(model_path, img_path, confidence):
    img = Image.open(img_path)
    mTransform = T.Compose([T.ToTensor()])
    img = mTransform(img)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    img = img.to(device)
    model = torch.load(model_path)
    predictions = model([img])
    predict_score = list(predictions[0]['scores'].detach().cpu().numpy())
    predict_t = [predict_score.index(x) for x in predict_score if x > confidence][-1]
    masks = (predictions[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    predict_class = [CLASS_NAMES[i] for i in list(predictions[0]['labels'].cpu().numpy())]
    predict_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:predict_t + 1]
    predict_boxes = predict_boxes[:predict_t + 1]
    predict_class = predict_class[:predict_t + 1]
    return masks, predict_boxes, predict_class


def segmentation_presenter(model_path, img_path, confidence=0.7):
    masks, boxes, predict_classes = predictor(model_path, img_path, confidence)
    img = cv2.imread(img_path)
    for i in range(len(masks)):
        rgb_mask = get_coloured_mask(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        x1 = int(boxes[i][0][0])
        y1 = int(boxes[i][0][1])
        x2 = int(boxes[i][1][0])
        y2 = int(boxes[i][1][1])
        cv2.rectangle(img, (x1,y1), (x2,y2), color=(0, 255, 0), thickness=3)
        cv2.putText(img, predict_classes[i], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), thickness=1)
        cv2.imshow('MaskRCNN', img)
        cv2.waitKey(0)


if __name__ == "__main__":
    CLASS_NAMES = ['__background__', 'pedestrian']
    path_model = '../Trained_Models/MaskRCNN_Models/maskRCNN.pt'
    path_test_image = "../Images_General_Test/images.jpg"
    segmentation_presenter(path_model, path_test_image)
