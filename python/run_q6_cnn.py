import numpy as np
import scipy.io
import torch
import torchvision.datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
#from run_q2 import *
# print(max_iters) # 500

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1,8,kernel_size=3,stride=1), #30
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2), #15
            nn.Conv2d(8,16,kernel_size=3,stride=1), #13
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Flatten())
        
        self.fc1 = nn.Linear(13*13*16, 48)
        self.s1 = nn.Sigmoid()
        self.fc2 = nn.Linear(48, 36)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        x = self.s1(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    np.random.seed(1)
    device = torch.device("cpu") #torch not installed with cuda enabled
    print("device =", device)   

    train_data = scipy.io.loadmat('data/nist36_train.mat')
    valid_data = scipy.io.loadmat('data/nist36_valid.mat')  

    train_x = train_data['train_data'].astype(np.float32)
    #print(train_x.shape)
    train_x = np.reshape(train_x,(train_x.shape[0],1,32,32))
    #print(train_x.shape)
    train_y = train_data['train_labels'].astype(np.int32)
    #print(train_y.shape)
    valid_x = valid_data['valid_data'].astype(np.float32)
    valid_x = np.reshape(valid_x,(valid_x.shape[0],1,32,32))
    valid_y = valid_data['valid_labels'].astype(np.int32)


    epochs = 50 # same as q2
    bs = 5 # same as q2
    model = Net().to(device)
    ### print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    train_data_tensor = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    test_data_tensor = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
    train_loader = DataLoader(train_data_tensor, batch_size=bs, shuffle=True, num_workers=1)
    valid_loader = DataLoader(test_data_tensor, batch_size=bs, shuffle=True, num_workers=1)

    model.train()
    acc = []
    losses = []
    for itr in range(epochs):
        correct = 0
        total_loss = 0
        count = 0
        for idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            target = torch.max(target, 1)[1]
            loss = criterion(output, target)            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
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