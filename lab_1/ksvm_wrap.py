import sklearn.svm as svm
import numpy as np
import data
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c = 1, param_svm_gamma = 'auto'):
        self.model = svm.SVC(C = param_svm_c, gamma=param_svm_gamma)
        self.model.fit(X, Y_)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_scores(self, X):
        return self.model.decision_function(X)
    
    def support(self):
        return self.model.support_
    
if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    model = KSVMWrap(X, Y_, param_svm_c=1, param_svm_gamma='auto')

    Y = model.predict(X)
    acc, pr, M = data.eval_perf_multi(Y, Y_)
    f1_macro = f1_score(Y_, Y, average='macro')
    print(f"Accuracy = {acc}, Precision = {pr}, F1 score = {f1_macro}")
    print(f"Confusion matrix =\n{M}")

    decfun = model.get_scores
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y, special=model.support())
    plt.show()
