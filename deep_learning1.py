# M: This program is a practise for defining a custom net model and use a pretrained model to train new data.
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
from efficientnet_pytorch import EfficientNet
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels,
        # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        # x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def getMeanStd(mTrain_data_dir,mTrain_batch_size,mTrain_num_workers):
    mTransform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor()])
    mTrain_data = ImageFolder(mTrain_data_dir, mTransform)
    mTrain_loader = DataLoader(mTrain_data, batch_size=mTrain_batch_size, shuffle=True, num_workers=mTrain_num_workers)
    mean = 0.0
    for images, _ in mTrain_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(mTrain_loader.dataset)

    var = 0.0
    for images, _ in mTrain_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (len(mTrain_loader.dataset) * 224 * 224))
    print(mean,std)
    return mean, std


def img_shower(img):
    img = img / 2 + 0.4511
    img_numpy = img.numpy()
    plt.imshow(np.transpose(img_numpy, (1, 2, 0)))
    plt.show()


def net_trainer(batch_times, net, path):
    startTime = time.perf_counter()

    learningRate = 0.001
    momentum = 0.9

    if torch.cuda.is_available():
        device = 'cuda'
        net.to(device)
    else:
        device = 'CPU'

    optimizer = optim.SGD(net.parameters(), learningRate, momentum)
    criterion = nn.CrossEntropyLoss()
    min_valid_loss = np.inf
    overfitting_Arr = []
    for epoch in range(batch_times):
        total_train_loss, train_count = 0.0, 0
        net.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_count = train_count + 1
            total_train_loss += loss.item() * data.size(0)
            print("M:", "Epoch:", epoch, " Count:", train_count, ' Tra_loss:', loss.item() * data.size(0))

        total_valid_loss, valid_count = 0.0, 0
        net.eval()
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            target = net(data)
            loss = criterion(target, labels)
            valid_count = valid_count + 1
            total_valid_loss += loss.item() * data.size(0)  # data.size(0) is the batch size
            # M: something is fancy
            mString1 = "M:"+"Epoch:"+str(epoch)+ " Count:"+str(valid_count)+ ' Val_loss:'+str(loss.item() * data.size(0))
            print(''.join(str(ord(c)) for c in mString1))

        print("M: Current loss: ", min_valid_loss, '\n',
              '   Total Tra_loss: ', total_train_loss, '\n',
              '   Total Val_Loss:', total_valid_loss, '\n',)

        if total_valid_loss < min_valid_loss:
            min_valid_loss = total_valid_loss
            torch.save(net.state_dict(), path)
        else:
            overfitting_Arr.append(epoch)
            print('M: Epoch:', epoch, "'s parameters are skipped due to the over-fitting")

    endTime = time.perf_counter()
    print('M: Over-fitting: ', overfitting_Arr)
    print("M: Training process took", f"{endTime - startTime:0.4f} Seconds.")


def net_tester(net):
    wrong_count = 0
    mIndex = 0
    bad_prediction_collection = []
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            mIndex = mIndex + 1
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            if labels != predicted:
                wrong_count = wrong_count + 1
                bad_prediction_collection.append(torchvision.utils.make_grid(images))
                print('M: Index:', mIndex, ' prediction:', predicted, ' Truth:', labels, wrong_count)
            else:
                print('M: Index:', mIndex, )
        result = 100 * (mIndex - wrong_count) / mIndex
        print('Accuracy on the', mIndex, 'number of images is: ', result)

    # M: present the wrong prediction
    for i in bad_prediction_collection:
        img_shower(i)


if __name__ == '__main__':
    # M: set parameters for batch_size and num_workers
    mTrain_batch_size = 128  #128
    mTrain_num_workers = 8
    test_batch_size = 1
    test_num_workers = 2
    mean = (0.4886, 0.4511, 0.4123)
    std = (0.2589, 0.2510, 0.2517)

    # M: get mean and std value for the whole train dataset
    if mean == (0,0,0) or std == (0,0,0):
        mean, std = getMeanStd("../Images_Dog_Cat_Test/train", mTrain_batch_size, mTrain_num_workers)

    # M: define the format of the data
    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    # M: pumping the data
    classes = ('cat', 'dog')
    train_data_dir = "../Images_Dog_Cat_Test/train"
    test_data_dir = "../Images_Dog_Cat_Test/test"
    train_data = ImageFolder(train_data_dir, transform)
    test_data = ImageFolder(test_data_dir, transform)
    train_loader = DataLoader(train_data, batch_size=mTrain_batch_size, shuffle=True, num_workers=mTrain_num_workers)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True, num_workers=test_num_workers)

    # M: define different net models
    net2 = models.alexnet(True)
    net3 = models.resnet101(pretrained=True)
    net4 = EfficientNet.from_pretrained('efficientnet-b0')

    # M: set the purpose of this program
    purpose1 = "train a new model"
    purpose2 = "test the saved models"
    whatYouWantIs = purpose1

    if whatYouWantIs == purpose1:
        net_trainer(10, net2, 'Trained_Models/alex_net3_15.pth')
        # net_trainer(15, net3, '../Trained_Models/res_net_15.pth')
        # net_trainer(15, net4, '../Trained_Models/eff_net_15.pth')
    elif whatYouWantIs == purpose2:
        path = '../Trained_Models/Eff_Models/eff_net_15.pth'
        net_shell = EfficientNet.from_name('efficientnet-b0').eval()
        # net_shell = models.resnet101(False).eval()  # M: this line of code needs to be checked.
        net_shell.load_state_dict(torch.load(path))
        loadedNet = net_shell
        net_tester(loadedNet)