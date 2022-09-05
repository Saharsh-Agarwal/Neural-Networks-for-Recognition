import numpy as np
import scipy.io
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=5,stride=1), #28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2), #14
            nn.Conv2d(16,16,kernel_size=5,stride=1), #10
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Flatten())
        
        self.fc1 = nn.Linear(10*10*16, 64)
        self.s1 = nn.Sigmoid()
        self.fc2 = nn.Linear(64, 32)
        self.s2 = nn.Sigmoid()
        self.fc3 = nn.Linear(32,10)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        x = self.s1(x)
        x = self.fc2(x)
        x = self.s2(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    np.random.seed(1)
    device = torch.device("cpu") #torch not installed with cuda enabled
    print("device =", device)   

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    epochs = 20
    bs = batch_size
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()
    acc = []
    losses = []
    for itr in range(epochs):
        print(itr)
        correct = 0
        total_loss = 0
        count = 0
        for idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            #target = torch.max(target, 1)[1]
            loss = criterion(output, target)            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            pred = torch.max(output, 1)[1] 
            correct += pred.eq(target).sum().item()
            count = count + 1
        
        acc.append(100. * correct / (count*bs))
        losses.append(total_loss/count)
        if itr % 2 == 0:
            print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,(total_loss/count),(100. * correct / (count*bs))))

    plt.figure()
    a = [i for i in acc]
    plt.plot(np.arange(epochs), a, label = "Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracies (%)")
    plt.title("Average Accuracies (%) vs Epochs")
    plt.show()

    plt.figure()
    l = [i for i in losses]
    plt.plot(np.arange(epochs), l, label = "Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("Average Loss vs Epochs")
    plt.show()  