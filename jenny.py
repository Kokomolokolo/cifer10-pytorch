# Convoluted neural network
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_config():
    return {
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
    }

# normalisieren der reichweite
def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def load_dataset(transform):
    train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            transform=transform,
                                            download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            transform=transform,
                                            download=True)
    return train_dataset, test_dataset
def get_data_loaders(train_dataset, test_dataset, batch_size):
    # data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

    return train_loader, test_loader
classes =('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(imgs):
    imgs = imgs / 2 + 0.5
    npimgs = imgs.numpy()
    plt.imshow(np.transpose(npimgs, (1, 2, 0)))
    plt.show()
def show_sample_data(train_loader):
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    img_grid = torchvision.utils.make_grid(images[0:25], nrow=5)
    imshow(img_grid)

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
    
def create_model(device):
    model = Jenny().to(device)
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train() # setzt das model wohl in den trainingszustand, wie das vorher gegangen ist idk
    running_loss = 0.0
    start_time = time.time()
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
    epoch_time = time.time() - start_time
    avg_loss = running_loss / len(train_loader)
    return avg_loss, epoch_time

def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    print('Started Training')
    print("-" * 60)

    total_start_time = time.time()

    for epoch in range(num_epochs):
        avg_loss, epoch_time = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.3f}, Time: {epoch_time:.3f}")
    
    total_time = time.time()- total_start_time 

    print("-" * 60)
    print(f'Finished Training in {total_time:.2f}s ({total_time/60:.2f} min)')
    print(f'Average time per epoch: {total_time/num_epochs:.2f}s')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using {device}")

    config = get_config()

    transform = get_transforms()
    train_dataset, test_dataset = load_dataset(transform)
    train_loader, test_loader = get_data_loaders(train_dataset, test_dataset, config["batch_size"])

    model = create_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_model(model, train_loader, criterion, optimizer, device, config["num_epochs"])

    PATH = "./jenny.pth"
    print("Saving model to {PATH}")
    torch.save(model.state_dict(), PATH)

    eval_model(test_loader, device)

def eval_model(test_loader, device):
    model = load_model(device)
    model.eval()

    # Eval Loop
    with torch.no_grad():
        n_corr = 0
        n_samples = len(test_loader.dataset)

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            
            _, predicted = torch.max(outputs, 1) # max wert der zeile, dim 1
            n_corr += (predicted == labels).sum().item()

        acc = 100.0 * n_corr / n_samples

        print(f"Das trainierte Model hat eine acc von {acc}")

def load_model(device):
    loaded_model = Jenny()
    loaded_model.load_state_dict(torch.load('jenny.pth', weights_only=False))
    loaded_model.to(device)
    return loaded_model

if __name__ == "__main__":
    main()