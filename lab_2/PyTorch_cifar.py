import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import skimage.io as ski

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding='same')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(32*7*7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.maxpool(torch.relu(self.conv1(x)))
        x = self.maxpool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[0]
    num_channels = w.shape[1]
    k = w.shape[2]
    assert w.shape[3] == w.shape[2]
    w = w.transpose(2, 3, 1, 0)
    w -= w.min()
    w /= w.max()
    cols = 8
    border = 1
    rows = int(np.ceil(num_filters / cols))
    height = rows * k + (rows - 1) * border
    width = cols * k + (cols - 1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k, c:c+k, :] = w[:, :, :, i]

    os.makedirs(save_dir, exist_ok=True)
    filename = f"epoch_{epoch:02d}_step_{step:06d}.png"
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    img = (img * 255).astype(np.uint8)
    ski.imsave(f"{save_dir}/{filename}", img)


def evaluate(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for X, Yoh_ in dataloader:
            Y_ = model(X)
            _, preds = torch.max(Y_, 1)
            all_labels.extend(Yoh_.numpy())
            all_preds.extend(preds.numpy())

    CM = confusion_matrix(all_labels, all_preds)
    print(f"Confusion matrix:\n{CM}")
    acc = (np.array(all_labels) == np.array(all_preds))
    acc = acc.mean()
    print(f"Accuracy = {acc * 100:.2f}%")
    pr = precision_score(all_labels, all_preds, average=None, labels=np.unique(all_labels))
    print(f"Precision = {pr}")
    rc = recall_score(all_labels, all_preds, average=None, labels=np.unique(all_labels))
    print(f"Recall = {rc}")

    return CM, acc, pr, rc

def train(model, trainloader, validloader, num_epochs=10, lr=0.001, save_dir='./trainig_results'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    plot_data = {
        'train_loss': [], 
        'valid_loss': [], 
        'train_acc': [], 
        'valid_acc': [], 
        'lr': []
        }

    draw_conv_filters(0, 0, model.conv1.weight.detach().numpy(), save_dir)

    for e in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (X, Yoh_) in enumerate(trainloader):
            Y_ = model(X)
            loss = criterion(Y_, Yoh_)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            _, preds = torch.max(Y_, 1)
            correct_train += (preds == Yoh_).sum().item()
            total_train += Yoh_.size(0)

            if i % 200 == 0:
                print(f"Epoch [{e + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / len(trainloader)
        train_acc = correct_train / total_train
        print(f"Epoch {e + 1}: Train loss: {train_loss:.4f}, Train accuracy: {train_acc * 100:.2f}%")

        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0
        model.eval()
        with torch.no_grad():
            for X, Yoh_ in validloader:
                Y_ = model(X)
                loss = criterion(Y_, Yoh_)
                valid_loss += loss.item()
                _, preds = torch.max(Y_, 1)
                correct_valid += (preds == Yoh_).sum().item()
                total_valid += Yoh_.size(0)

        valid_loss /= len(validloader)
        valid_acc = correct_valid / total_valid
        print(f"Epoch {e + 1}: Validation loss: {valid_loss:.4f}, Validation accuracy: {valid_acc * 100:.2f}%")

        plot_data['train_loss'].append(train_loss)
        plot_data['valid_loss'].append(valid_loss)
        plot_data['train_acc'].append(train_acc)
        plot_data['valid_acc'].append(valid_acc)
        plot_data['lr'].append(optimizer.param_groups[0]['lr'])

        weights = model.conv1.weight.detach().numpy()
        draw_conv_filters(e + 1, 0, weights, save_dir)
    return plot_data
SAVE_DIR = './training_results'

def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 7))
    epochs = list(range(1, len(data['train_loss']) + 1))

    tr_color = 'r'
    val_color = 'b'

    ax1.plot(data['train_loss'], label='Train Loss', color=tr_color)
    ax1.plot(data['valid_loss'], label='Validation Loss', color=val_color)
    ax1.set_title('Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(data['train_acc'], label='Train Accuracy', color=tr_color)
    ax2.plot(data['valid_acc'], label='Validation Accuracy', color=val_color)
    ax2.set_title('Average Class Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax2.legend(loc='best')

    ax3.plot(data['lr'], label='Learning Rate')
    ax3.set_title('Learning rate')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate')

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/training_plot.png")
    plt.show()

if __name__ == '__main__':
    save_dir = './training_results'
    lr = 0.001
    num_epochs = 10

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=50, shuffle=True)
    test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test_ds, batch_size=50, shuffle=False)

    model = ConvNet()
    plot_data = train(model, trainloader, testloader, num_epochs, lr, save_dir)
    plot_training_progress(save_dir, plot_data)
    print(f"Evaluation on Test set:")
    evaluate(model, testloader)
