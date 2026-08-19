import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

def generate_data(a, b, n, sigma):
    X = torch.randn(n)
    Y = a * X + b + sigma * torch.randn(n)
    return X, Y

X, Y = generate_data(a = -2, b = 3, n = 50, sigma = 0.5)
optimizer = optim.SGD([a, b], lr=0.1)

for i in range(100):
    Y_ = a*X + b
    diff = (Y-Y_)

    loss = torch.mean(diff**2)
    loss.backward()

    print(f"Step {i}: loss = {loss.item()}, grad_a = {a.grad}, grad_b = {b.grad}")

    grad_a = torch.mean(-2 * X * diff)
    grad_b = torch.mean(-2 * diff)
    print(f"Analytical grad_a = {grad_a}, Analytical grad_b = {grad_b}")

    assert torch.isclose(a.grad, grad_a, atol=1e-6), "Gradients for a do not match"
    assert torch.isclose(b.grad, grad_b, atol=1e-6), "Gradients for b do not match"

    optimizer.step()
    optimizer.zero_grad()

plt.plot(X.detach().numpy(), Y.detach().numpy(), 'o', label='$(x^{(i)},y^{(i)})$')
plt.plot(X.detach().numpy(), (a * X + b).detach().numpy(), label='$\\hat{y} = ax + b$', color='r')

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()