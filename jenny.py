# Convoluted neural network
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 10
batch_size = 32
learning_rate = 0.001

# normalisieren der reichweite
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                          train=True,
                                          transform=transform,
                                          download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                          train=False,
                                          transform=transform,
                                          download=True)

# data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

classes =('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(imgs):
    imgs = imgs / 2 + 0.5
    npimgs = imgs.numpy()
    plt.imshow(np.transpose(npimgs, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = next(dataiter)
img_grid = torchvision.utils.make_grid(images[0:25], nrow=5)
# imshow(img_grid)

class Jenny(nn.Module):
    def __init__(self,):
        super(Jenny, self).__init__()
        # Conv habe ich so halb verstanden. Es nimmt sich einen kernel, hier 3x3, und schiebt ihn über das bild. Das hilft beim erkennen
        # von genaueren Konturen. Mehrere Conv erarbeiten immer genauere Unterschiede herraus. Aus gibt es eine Unabhänigkeit von dem
        # Ort des Objektes im Bild.
        self.conv1 = nn.Conv2d(3, 32, 3) # 3 colo channels, output, kernel size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64*4*4, 64) # steht für fully connected
        self.fc2 = nn.Linear(64, 10) # 10 wegen 10 output classes

    def forward(self, x):
        # N = inputsize, 3 color chanels, 32, 32 bildgröße
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = f.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = Jenny().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backwards
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
    print(f"{epoch+1}, loss: {running_loss / n_total_steps:.3f}")
print('Finished Training')
PATH = './jenny.pth'
torch.save(model.state_dict(), PATH)
