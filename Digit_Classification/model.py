import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import os.path
from PIL import Image, ImageOps
import argparse

def load_data():
    transform = transforms.Compose(
        [transforms.Resize((28, 28)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    # testset = torchvision.datasets.MNIST(root='./data', train=False,
    #                                      download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4,
    #                                          shuffle=False, num_workers=2)

    classes = ('0', '1', '2', '3',
               '4', '5', '6', '7', '8', '9')
    return(trainloader)

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

class Net(nn.Module):
    def __init__(self):
        # This nn has 4 hidden layers and 1 output layer. Of the 4 hidden layers, there are two convolutional
        # and two linear (Feed Forward?)

        super(Net, self).__init__()
        # 3 input channels, 6 output channels, 5 kernel size
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # 6 input channels, 16 output channels, 5 kernel size
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(4 * 4 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #         print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        #         print(x.shape)
        # Reshape the tensor to the correct dimension. Here '-1' means that we do not know how many rows there are.
        # System will compute the correct number of rows based on the column requirements we have given
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(trainloader):
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                # print('[%d, %5d] loss: %.3f' %
                #       (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


def save_model(path=None):
    if path is None:
        path = "/Users/balakumaranpanneerselvam/Documents/Bala/projects/torch/Digit_Classification/model_saved.pt"
    torch.save(net.state_dict(), path)


def load_model(path=None):
    if path is None:
        path = "/Users/balakumaranpanneerselvam/Documents/Bala/projects/torch/Digit_Classification/model_saved.pt"

    if not os.path.isfile(path):
        trainloader = load_data()
        net = Net()
    net = Net()
    net.load_state_dict(torch.load(path))
    net.eval()
    return net


if __name__ == "__main__":
    trainloader = load_data()
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train(trainloader)
    save_model()
    net = load_model()
