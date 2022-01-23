#M: This program is a practise to experiement pretrained models.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import cv2
from efficientnet_pytorch import EfficientNet

# M: create models
alexnet = models.AlexNet()
resnet = models.resnet101(pretrained=True)
effnet = EfficientNet.from_pretrained('efficientnet-b0')

# M: preprocess images
mean = (0.4886, 0.4511, 0.4123)
std = (0.2589, 0.2510, 0.2517)
preprocess = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean, std)])

# M: read the image
fileName = "../Images_General_Test/golden_retriever.png"
img = Image.open(fileName).convert('RGB')
# img.show()

# M: prepare the label for matching
with open('../Images_General_Test/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t,0)
effnet.eval()
out = effnet(batch_t)
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
_, indices = torch.sort(out, descending=True)
count = 0
for idx in indices[0][:5]:
    count = count + 1
    result = "M: " + str(count) + " " + labels[idx] + " " + str(percentage[idx].item())
    print(result)




