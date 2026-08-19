import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import data
from itertools import product
from sklearn.metrics import f1_score

class PTLogreg(nn.Module):
  def __init__(self, D, C):
    """Arguments:
       - D: dimensions of each datapoint 
       - C: number of classes
    """
    super(PTLogreg, self).__init__()

    self.W = nn.Parameter(torch.randn(D, C), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(C), requires_grad=True)

  def forward(self, X):
    s = torch.mm(X, self.W) + self.b
    return torch.softmax(s, dim=1)

  def get_loss(self, X, Yoh_):
    probs = self.forward(X)
    return -torch.sum(Yoh_ * torch.log(probs + 1e-20), dim=1).mean()


def train(model, X, Yoh_, param_niter, param_delta, param_lambda):
  """Arguments:
     - X: model inputs [NxD], type: torch.Tensor
     - Yoh_: ground truth [NxC], type: torch.Tensor
     - param_niter: number of training iterations
     - param_delta: learning rate
  """
  model.train()
  optimizer = optim.SGD(model.parameters(), lr = param_delta)
  for i in range(int(param_niter)):
    loss = model.get_loss(X, Yoh_) + param_lambda * torch.norm(model.W)
    loss.backward()

    if i % 100 == 0:
      print(f"Iteration {i}: loss = {loss}")

    optimizer.step()
    optimizer.zero_grad()

  return


def eval(model, X):
  """Arguments:
     - model: type: PTLogreg
     - X: actual datapoints [NxD], type: np.array
     Returns: predicted class probabilites [NxC], type: np.array
  """
  X = torch.Tensor(X)
  Y = model.forward(X)
  return Y.detach().numpy()

def logreg_decfun(model):
    return lambda X: np.argmax(eval(model, X), axis=1)

if __name__ == "__main__":
  np.random.seed(100)

  X, Y = data.sample_gauss_2d(3, 100)
  X = torch.Tensor(X)
  Yoh_ = data.class_to_onehot(Y)
  Yoh_ = torch.Tensor(Yoh_)

  niter_values = [1000]
  delta_values = [0.5]
  lambda_values = [0.05]

  print(f"Hyperparameters: niter = {niter_values}, delta = {delta_values}, lambda = {lambda_values}")

  accurs = {}

  for n, d, l in product(niter_values, delta_values, lambda_values):
    
    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

    train(ptlr, X, Yoh_, param_niter=n, param_delta=d, param_lambda=l)
    
    probs = eval(ptlr, X)
    Y_pred = np.argmax(probs, axis=1)
    acc, pr, M = data.eval_perf_multi(Y, Y_pred)
    f1_macro = f1_score(Y, Y_pred, average='macro')

    accurs[(n, d, l)] = acc
    print(f"Accuracy: {acc}, F1 score: {f1_macro}") 
    
    X_ = X.detach().numpy()
    decfun = logreg_decfun(ptlr)
    bbox=(np.min(X_, axis=0), np.max(X_, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X_, Y_pred, Y)
    plt.show()
