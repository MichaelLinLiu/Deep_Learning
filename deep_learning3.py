# M: This program aims to train net with different number of batches and save the tested result.
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import csv
from datetime import datetime
from efficientnet_pytorch import EfficientNet


def net_trainer(batch_times, net, net_shell, path, modelName):
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
    csvRows = []
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
            print(" " * valid_count + "\U0001F3C3")

        print("M: Current loss: ", min_valid_loss, '\n',
              '   Total Tra_loss: ', total_train_loss, '\n',
              '   Total Val_Loss:', total_valid_loss, '\n',)

        net_shell.load_state_dict(net.state_dict(), strict=False)
        loadedNet = net_shell
        accuracy = net_tester(loadedNet)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        # modelFileName = modelName+str(epoch)+'.pt'
        csvRows.append([current_time, modelName, str(epoch+1), int(total_train_loss), int(total_valid_loss),str(accuracy)])
        fields = ['Time', 'Model Name', 'Epoch', 'Train Loss', 'Valid Loss', 'Accuracy']
        csvName = modelName+'.csv'
        csvPath = path + csvName
        with open(csvPath, 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(fields)
            writer.writerows(csvRows)
        csvFile.close()

    endTime = time.perf_counter()
    print("M: Training process took", f"{endTime - startTime:0.4f} Seconds.")


def net_tester(net):
    if torch.cuda.is_available():
        device = 'cuda'
        net.to(device)
    else:
        device = 'CPU'
    wrong_count = 0
    mIndex = 0
    net.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            mIndex = mIndex + 1
            outputs = net(data)
            _, predicted = torch.max(outputs.data, 1)
            if labels != predicted:
                wrong_count = wrong_count + 1
                print('M: Index:', mIndex, ' prediction:', predicted, ' Truth:', labels, wrong_count)
            else:
                print('M: Index:', mIndex, )
        result = 100 * (mIndex - wrong_count) / mIndex
        print('Accuracy on the', mIndex, 'number of images is: ', result)
    return result


if __name__ == '__main__':
    # M: set parameters for batch_size and num_workers
    mTrain_batch_size = 64 #128 #24
    mTrain_num_workers = 8
    test_batch_size = 1
    test_num_workers = 2
    mean = (0.4886, 0.4511, 0.4123)
    std = (0.2589, 0.2510, 0.2517)

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
    net2_1 = models.alexnet(False)
    net3 = models.resnet101(True)
    net3_1 = models.resnet101(False)
    net4 = EfficientNet.from_pretrained('efficientnet-b0')
    net4_1 = EfficientNet.from_name('efficientnet-b0')
    net5 = models.googlenet(True)
    net5_1 = models.googlenet(False)
    net6 = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    net6_1 = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)

    # M: start cooking
    # net_trainer(50, net2, net2_1, "../Trained_Models/Alex_Models/", 'AlexNet')
    # net_trainer(50, net3, net3_1, "../Trained_Models/Res_Models/", 'ResNet')
    # net_trainer(50, net4, net4_1, "../Trained_Models/Eff_Models/", 'EfficientNet')
    # net_trainer(50, net5, net5_1, "../Trained_Models/Goo_Models/", 'GoogleNet')
    net_trainer(50, net6, net6_1, "../Trained_Models/Mobi_Models/", 'MobileNet')
