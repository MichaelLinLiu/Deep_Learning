import labelme2coco
import cv2
import numpy as np
from PIL import Image
import torch
import transforms as mT

# labelme_folder = "./Images_Segmentation_Train"
#
# # set path for coco json to be saved
# save_json_path = "../test_coco.json"
#
# # conert labelme annotations to coco
# labelme2coco.convert(labelme_folder, save_json_path)

def get_transform(train):
    tTransforms = [mT.ToTensor()]
    if train:
        tTransforms.append(mT.RandomHorizontalFlip(0.5))
    return mT.Compose(tTransforms)


# fileName = '../Images_Segmentation_Test/PNGImages/FudanPed00006.png'
# img = cv2.imread(fileName, 1)
# cv2.imshow('Original Image', img)

fileName2 = '../Images_Segmentation_Test/PedMasks/FudanPed00006_mask.png'
# mask = cv2.imread(fileName2, 1)
# height = mask.shape[0]
# width = mask.shape[1]
# result = np.zeros((height, width), dtype=np.uint8)
# for h in range(height):
#     for w in range(width):
#         newValue = 0
#         if mask[h,w,0] == 1:
#             newValue = 128
#         elif mask[h,w,0] == 2:
#             newValue == 0
#         else:
#             newValue = 255
#         result[h,w] = newValue
# cv2.imshow('masked image',result)
# cv2.waitKey(0)


mask = Image.open(fileName2)
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
idx = 10
image_id = torch.tensor([idx])
area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
isCrowd = torch.zeros((num_objects,), dtype=torch.int64)
target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area,
          "isCrowd": isCrowd}
print(target)
cv2.imshow('masked image', mask)
cv2.waitKey(0)

# M: convert to tensors
transforms = get_transform(train=True)
if transforms is not None:
    img, target = transforms(mask, target)




