import numpy as np
import data
import matplotlib.pyplot as plt

def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return exp_x / exp_x.sum(axis=1).reshape(-1, 1)

class FCANN2:
    def __init__(self, input_dim = 2, output_dim = 2, hidden_dim = 5, param_lambda = 1e-3, param_delta = 0.05, param_niter = (1e5)):
        self.W1 = np.random.normal(scale = 1 / np.sqrt(input_dim), size= (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.normal(scale = 1 / np.sqrt(hidden_dim), size= (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

        self.param_lambda = param_lambda
        self.param_delta = param_delta
        self.param_niter = param_niter
    

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = ReLU(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = softmax(self.z2)
        return self.a2
    
    def train(self, X, Y_):
        N, D = X.shape
        C = np.max(Y_) + 1
        Y_onehot = np.zeros((N, C))
        Y_onehot[np.arange(N), Y_] = 1

        for i in range(int(self.param_niter)):
            probs = self.forward(X)
            loss = -np.sum(Y_onehot * np.log(probs + 1e-20)) / N + self.param_lambda * (np.sum(self.W1**2) + np.sum(self.W2**2))

            if i % (self.param_niter // 10) == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")

            dscores = probs - Y_onehot
            dscores = dscores / N

            dW2 = self.a1.T @ dscores + self.param_lambda * self.W2
            db2 = np.sum(dscores, axis=0)

            dH = dscores @ self.W2.T
            dH[self.a1 <= 0] = 0

            dW1 = X.T @ dH + self.param_lambda * self.W1
            db1 = np.sum(dH, axis=0)

            self.W1 = self.W1 - self.param_delta * dW1
            self.b1 = self.b1 - self.param_delta * db1
            self.W2 = self.W2 - self.param_delta * dW2
            self.b2 = self.b2 - self.param_delta * db2

    def classify(self, X):
        return np.argmax(self.forward(X), axis=1)
    
def fcann2_train(X, Y_):
    model = FCANN2()
    model.train(X, Y_)
    return model

def fcann2_classify(X, model):
    return model.classify(X)

def fcann2_decfun(model):
    def decision_function(X):
        return fcann2_classify(X, model)
    return decision_function

if __name__ == "__main__":
    np.random.seed(100)
    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    model = fcann2_train(X, Y_)
    Y = fcann2_classify(X, model)
    accuracy, precision, CM = data.eval_perf_multi(Y, Y_)
    print(f"Accuracy = {accuracy}\n Precision = {precision}\n Confusion matrix:\n{CM}")
    decfun = fcann2_decfun(model)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y)
    plt.show()