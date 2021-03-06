import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.model_selection import KFold
# from skorch import NeuralNetClassifier
from torch.utils.data import Subset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         shuffle=False, num_workers=2)

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


criterion = nn.CrossEntropyLoss()
kf = KFold(n_splits=5)
max_epochs = 5

accuracies = []

for lr in [0.001, 0.0005, 0.0001]:
    for idx in range(0, 4):

        partAcc = []
        for train_index, val_index in kf.split(trainset):
            #X_train, X_test = X[train_index], X[test_index]
            #y_train, y_test = y[train_index], y[test_index]
            train_split = Subset(trainset, train_index)
            val_split = Subset(trainset, val_index)

            trainloader = torch.utils.data.DataLoader(train_split, batch_size=4,
                                                      shuffle=True, num_workers=2)

            valloader = torch.utils.data.DataLoader(val_split, batch_size=4,
                                                      shuffle=True, num_workers=2)

            running_loss = 0.0
            running_acc = 0.0
            val_acc = 0.0
            total = 0


            net = AlexNet()
            net.cuda()

            if idx == 0:
                optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
            elif idx == 1:
                optimizer = optim.RMSprop(net.parameters(), lr=lr)
            elif idx == 2:
                optimizer = optim.Adam(net.parameters(), lr=lr)
            else:
                optimizer = optim.Adamax(net.parameters(), lr=lr)


            for epoch in range(max_epochs):
                for i, data in enumerate(trainloader, 0):
                    # get the inputs
                    inputs, labels = data[0].to(device), data[1].to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                            # print statistics
                    running_loss += loss.item()
                    running_acc += (predicted == labels).sum().item()
                    total += labels.size(0)

                    if i % 2000 == 1999:    # print every 2000 mini-batches
                        print('minibatch: %5d loss: %.3f accuracy: %.3f' %
                              (i + 1, running_loss / 2000, 100 * running_acc / total))
                        running_loss = 0.0
                        running_acc = 0.0
                        total = 0


            print('Finished Training')

            correct = 0
            total = 0
            with torch.no_grad():
                for data in valloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = (100 * correct / total)
            partAcc.append(val_accuracy)
            print(partAcc)
            print('Accuracy of the network on the validation split: %.3f %%' % val_accuracy)

        print('n: %d' % idx)
        accuracies.append(partAcc)

print(accuracies)

avg_accuracies = []
for acc_id in accuracies:
    avg_accuracies.append(sum(acc_id)/len(acc_id))


print(avg_accuracies)
# [[35.16, 53.2, 61.08, 67.98, 70.2, 73.16, 77.34, 83.26, 83.92, 87.52],
# [55.04, 62.62, 69.88, 73.68, 77.56, 82.84, 85.24, 88.12, 87.8, 91.86],
# [54.04, 60.94, 64.98, 67.36, 67.48, 69.2, 69.14, 73.3, 70.12, 73.26]]

# [[34.26, 52.06, 60.8, 64.3, 71.66, 76.64, 77.14, 83.28, 84.78, 86.2],
#  [40.92, 45.72, 47.72, 46.76, 50.86, 50.8, 49.56, 49.4, 51.82, 51.98],
#  [56.06, 62.68, 67.42, 72.72, 78.4, 81.74, 83.84, 87.62, 89.36, 93.04],
#  [41.6, 38.74, 44.26, 40.54, 42.32, 44.3, 38.46, 39.94, 45.02, 41.6],

#  [22.94, 37.68, 50.78, 61.06, 66.86, 69.32, 71.24, 76.96, 76.98, 83.08],
#  [45.44, 55.48, 59.22, 60.86, 64.26, 64.34, 65.72, 68.34, 68.3, 69.42],
#  [50.26, 62.3, 67.92, 74.36, 76.74, 82.56, 85.42, 88.28, 92.36, 93.84],
#  [48.82, 50.34, 54.2, 48.06, 52.9, 54.7, 55.24, 53.76, 48.12, 53.22],

#  [11.36, 9.04, 13.68, 17.52, 26.0, 29.88, 33.18, 39.8, 44.78, 47.46],
#  [51.98, 63.08, 65.78, 72.98, 75.56, 82.7, 85.24, 87.92, 89.6, 92.46],
#  [47.06, 54.64, 57.98, 63.4, 66.64, 71.68, 73.14, 75.6, 80.66, 83.32],
#  [54.72, 62.0, 64.24, 67.56, 68.24, 71.48, 72.64, 73.66, 73.02, 73.58]]

# [[72.59, 72.57, 72.23, 70.9, 72.94],
# [43.5, 35.53, 47.43, 39.2, 40.17],
# [47.57, 10.07, 48.71, 48.65, 10.16],
# [72.58, 72.68, 72.44, 72.36, 72.24],
#
# [71.89, 70.08, 70.82, 72.04, 70.08],
# [51.08, 52.1, 56.39, 48.63, 53.47],
# [65.17, 64.19, 62.61, 65.05, 65.48],
# [73.59, 71.64, 72.31, 72.27, 71.85],
#
# [46.42, 46.39, 44.54, 44.19, 44.52],
# [66.17, 66.2, 67.76, 65.08, 62.86],
# [71.44, 70.22, 70.43, 71.25, 70.41],
# [68.43, 68.64, 68.71, 68.02, 68.2]]

# [72.246,
# 41.16600000000001,
# 33.032,
# 72.46000000000001,
# 70.982,
# 52.33399999999999,
# 64.50000000000001,
# 72.332,
# 45.212,
# 65.614,
# 70.75,
# 68.39999999999999]

# [[65.57, 65.77, 66.98, 66.8, 70.43],
# [38.28, 45.5, 41.28, 40.32, 44.69],
# [47.07, 46.19, 48.69, 48.73, 49.11],
# [69.67, 71.0, 71.7, 69.36, 69.19],
#
# [61.34, 59.84, 61.39, 59.33, 62.31],
# [53.87, 48.39, 52.35, 55.93, 53.8],
# [60.94, 60.82, 62.96, 62.52, 56.79],
# [69.48, 69.39, 70.01, 70.19, 69.44],
#
# [22.76, 20.36, 21.02, 21.48, 19.05],
# [66.34, 61.46, 65.91, 64.22, 65.89],
# [69.69, 67.99, 69.93, 68.52, 69.58],
# [62.51, 62.95, 62.44, 61.43, 62.95]]

# [67.11,
# 42.013999999999996,
# 47.95799999999999,
# 70.184,
#
# 60.842,
# 52.867999999999995,
# 60.806000000000004,
# 69.702,
#
# 20.934,
# 64.764,
# 69.142,
# 62.456]






