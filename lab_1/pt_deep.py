import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import data
import matplotlib.pyplot as plt
from itertools import product

class PTDeep(nn.Module):
  def __init__(self, dims, activation_func=torch.relu):
    super(PTDeep, self).__init__()

    self.activation_func = activation_func

    self.weights = nn.ParameterList([nn.Parameter(torch.randn(dims[i], dims[i + 1])) for i in range(len(dims) - 1)])
    self.biases = nn.ParameterList([nn.Parameter(torch.zeros(dims[i + 1])) for i in range(len(dims) - 1)])

  def forward(self, X):
    h = X
    for i in range(len(self.weights) - 1):
      h = self.activation_func(torch.mm(h, self.weights[i]) + self.biases[i])
    return torch.softmax(torch.mm(h, self.weights[-1]) + self.biases[-1], dim=1)

  def get_loss(self, X, Yoh_):
    probs = self.forward(X)
    return -torch.sum(Yoh_ * torch.log(probs + 1e-10), dim=1).mean()
    
  def count_params(self):
    counter = 0
    for i, j in self.named_parameters():
        print(f"{i} : {j.shape}")
        counter += j.numel()
    print(f"Total number of parameters: {counter}")
    return counter

def train(model, train_dataloader, param_niter, param_delta, param_lambda):
  model.train()
  optimizer = optim.SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)

  for i in range(int(param_niter)):
    for X, Yoh_ in train_dataloader:
      loss = model.get_loss(X, Yoh_)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
    if i % 100 == 0:
      print(f"Iteration {i}: loss = {loss}")
  return


def eval(model, X):
  model.eval()
  X = torch.Tensor(X)
  return model.forward(X).detach().numpy()

def deep_decfun(model):
  return lambda X: np.argmax(eval(model, X), axis=1) 

if __name__ == "__main__":
  np.random.seed(100)

  param_niter = 10000
  param_delta = 0.1
  param_lambda = 1e-4
  configs = [[2, 2], [2, 10, 2], [2, 10, 10, 2]]
  activation_func = torch.relu

  for c in configs:
    print(f"Config: {c}")

    X, Y = data.sample_gmm_2d(6, 2, 10)
    X = torch.Tensor(X)
    Yoh_ = data.class_to_onehot(Y)
    Yoh_ = torch.Tensor(Yoh_)

    train_data = torch.utils.data.TensorDataset(X, Yoh_)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32)

    ptdeep = PTDeep(c, activation_func=torch.relu)
    ptdeep.count_params()
    train(ptdeep, train_dataloader, param_niter, param_delta, param_lambda)

    probs = eval(ptdeep, X)
    Y_pred = np.argmax(probs, axis=1)

    acc, pr, M = data.eval_perf_multi(Y_pred, Y)
    avg_precision = np.mean([p[0] for p in pr])
    recall = np.mean([p[1] for p in pr])
    f1_macro = np.mean([2 * (p[0] * p[1]) / (p[0] + p[1]) for p in pr])

    print(f"Accuracy = {acc}, Average precision = {avg_precision}, Recall = {recall}, F1 score = {f1_macro}")

    X_ = X.detach().numpy()
    decfun = deep_decfun(ptdeep)
    bbox = (np.min(X_, axis=0), np.max(X_,axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X_, Y_pred, Y)
    plt.show()
