import numpy as np
import data
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import f1_score

def batch(X, Y, batch_size=32):
    indices = np.arange(len(X))
    for i in range(0, len(X), batch_size):
        batch_indices = indices[i:i+batch_size]
        yield X[batch_indices], Y[batch_indices]

class FCNN(nn.Module):
    def __init__(self, layers):
        super(FCNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.layers.append(nn.ReLU())
        self.layers.append(nn.Softmax(dim=1))

    def forward(self, X):
        for l in self.layers:
            X = l(X)
        return X
    
def train(model, train_data, val_data, epochs=15, lr=1e-4, reg=1e-5):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    criterion = nn.CrossEntropyLoss()
    train_loss, val_loss = [], []

    X_train, Y_train = train_data
    X_val, Y_val = val_data

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, Y_batch in batch(X_train, Y_train):
            loss = criterion(model(X_batch), Y_batch)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
        train_loss.append(epoch_loss / len(X_train))

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss.append(criterion(val_output, Y_val).item())
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Train loss = {train_loss[-1]:.4f}, Validation loss = {val_loss[-1]:.4f}")

    return train_loss, val_loss

def evaluate(model, X_test, Y_test):
    model.eval()
    Y_ = model(X_test)
    loss = F.cross_entropy(Y_, Y_test)
    acc, pr, M = data.eval_perf_multi(np.argmax(Y_.detach().numpy(), axis=1), Y_test)
    f1_macro = f1_score(np.argmax(Y_.detach().numpy(), axis=1), Y_test.numpy(), average='macro')
    return loss.item(), acc, pr, M, f1_macro

if __name__ == '__main__':
    dataset_root = '/tmp/mnist'
    transform = torchvision.transforms.ToTensor()
    train_data = torchvision.datasets.MNIST(dataset_root, train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(dataset_root, train=False, download=True, transform=transform)

    train_size = int(0.8*len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

    X_train = train_data.dataset.data[train_data.indices].float().view(train_size, -1) / 255.0
    Y_train = train_data.dataset.targets[train_data.indices]
    X_test = test_data.data.float().view(len(test_data), -1) / 255.0
    Y_test = test_data.targets
    X_val = val_data.dataset.data[val_data.indices].float().view(val_size, -1) / 255.0
    Y_val = val_data.dataset.targets[val_data.indices]

    model = FCNN([784, 100, 10])
    train_loss, val_loss = train(model, (X_train, Y_train), (X_val, Y_val))

    test_loss, test_accuracy, test_precision, test_conf_matrix, test_f1 = evaluate(model, X_test, Y_test)
    print(f"Test accuracy = {test_accuracy*100:.2f}%")

    random_indices = np.random.choice(len(X_test), 10, replace=False)
    X_sample = X_test[random_indices]
    Y_sample = Y_test[random_indices]

    Y_pred = model(X_sample)
    Y_pred = torch.argmax(Y_pred, dim=1)

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_sample[i].view(28, 28), cmap='gray')
        ax.set_title(f'Pred: {Y_pred[i].item()}\nAct: {Y_sample[i].item()}')
        ax.axis('off')
    plt.show()
