# M: this program is a practice for semantic segmentation by using Mask R-CNN
import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
from PIL import Image
import torch
import transforms as mT
import torchvision
import numpy as np


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objects = len(obj_ids)
        boxes = []
        for i in range(num_objects):
            pos = np.where(masks[i])
            xMin = np.min(pos[1])
            xMax = np.max(pos[1])
            yMin = np.min(pos[0])
            yMax = np.max(pos[0])
            boxes.append([xMin, yMin, xMax, yMax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objects,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        isCrowd = torch.zeros((num_objects,), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area,
                  "isCrowd": isCrowd}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # M: take the input_features out from model
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # M: use FASTRCNN's box predictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # M: use MaskRCNN for mask predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def get_transform(train):
    tTransforms = [mT.ToTensor()]
    if train:
        tTransforms.append(mT.RandomHorizontalFlip(0.5))
    return mT.Compose(tTransforms)


def trainer(mDevice, mCLASS_NAMES, mPath_data, mPath_model):
    num_classes = len(mCLASS_NAMES)
    mModel = get_model_instance_segmentation(num_classes)
    mModel.to(mDevice)

    # M: organise data
    dataset = PennFudanDataset(mPath_data, get_transform(train=True))
    dataset_test = PennFudanDataset(mPath_data, get_transform(train=False))
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=2,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)

    # M: set optimizer and parameters
    params = [p for p in mModel.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(mModel, optimizer, data_loader, mDevice, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(mModel, data_loader_test, device=mDevice)

    torch.save(mModel, mPath_model)
    return mModel


if __name__ == "__main__":
    path_data = '../Images_Segmentation_Test/'
    path_model = '../Trained_Models/MaskRCNN_Models/maskRCNN.pt'
    CLASS_NAMES = ['__background__', 'pedestrian']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    trainer(device, CLASS_NAMES, path_data, path_model)
