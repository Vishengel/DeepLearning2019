import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.model_selection import KFold
# from skorch import NeuralNetClassifier
from torch.utils.data import Subset

# ToDo
# - Increase amount of training epochs
# - Re-init network after each fold
# - Choose optimizers
# - Automate optimizer comparison
# - Plot accuracy over time

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

net = LeNet()
net.cuda()
"""
skorch_net = NeuralNetClassifier(
    module=LeNet(),
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.SGD(net.parameters(), lr=0.001, momentum=0.9),
    batch_size=64,
    train_split=None)
"""

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=0.001)
optimizer = optim.RMSprop(net.parameters(), lr=0.001)

#X = []
#y = []

#for data in trainset:
#    X.append(data[0])
#    y.append(data[1])

kf = KFold(n_splits=25)
accuracies = []

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
    accuracies.append(val_accuracy)
    print('Accuracy of the network on the validation split: %d %%' % val_accuracy)



