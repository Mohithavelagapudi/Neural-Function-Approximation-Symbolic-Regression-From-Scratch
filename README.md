# Neural Function Approximation & Symbolic Regression From Scratch

**End-to-end implementation of linear, multilayer perceptron, and custom LSTM architectures (NumPy & PyTorch) for continuous function approximation with post‑hoc sparse symbolic extraction.**

---
## Abstract
This repository investigates the classical yet still practically relevant problem of deterministic continuous function approximation and interpretable model extraction. We implement neural architectures from first principles (no high‑level frameworks for core math) to approximate nonlinear target functions f: \mathbb{R} \to \mathbb{R}. Beyond predictive fidelity (low Mean Squared Error), we pursue interpretability via sparse symbolic regression (LASSO) on engineered analytic bases (polynomials, trigonometric, exponential). The work demonstrates: 
- (1) constructive derivation of forward and backward passes
- (2) optimization dynamics (SGD & Adam)
- (3) internal representation analysis (hidden activations, PCA manifolds), and
- (4) translation of distributed representations into closed-form surrogate equations.

---
## Motivation & Problem Statement
We investigate a hybrid pipeline: train lightweight neural approximators, then distill their learned mapping into compact symbolic forms using convex sparsity (L1). This yields:
- High-fidelity surrogate (low MSE across domain)
- Closed-form approximation for downstream reasoning, error bounding, and derivative access

Core questions:
1. How do parameter dynamics of a minimal single neuron recover linear mappings?  
2. How do deeper tanh MLPs capture composite functional structure (oscillatory + polynomial)?  
3. Can a sequence model (custom LSTM) learn scalar functions when inputs are recoded as digit sequences—probing representation learning and invariances?  
4. How effectively can sparse post-hoc regression recover interpretable basis combinations approximating the learned function?  

---
## Repository Structure
| Component | File | Description |
|-----------|------|-------------|
| Single linear neuron | `Single_NN.ipynb` | Derives MSE gradients analytically for y = w x + b and visualizes parameter convergence. |
| MLP + Symbolic pipeline | `Symbolic_Function_Approximation_with_Neural_Networks.ipynb` | Implements multi-layer tanh network, Adam optimizer, coordinate descent LASSO, and SymPy conversion. |
| Custom LSTM + symbolic distillation | `Symbolic_Function_Approximation_LSTM (1).ipynb` | Manual LSTM gate equations (PyTorch), sequence encoding of scalars, activation manifold analysis (PCA), linear basis fit. |

---
## Mathematical Foundations

<p align="center">
  <img src="images/image (39).png" alt="" width="1000"/>
</p>

---

<p align="center">
  <img src="images/image (40).png" alt="" width="1000"/>
</p>

---

<p align="center">
  <img src="images/image (41).png" alt="" width="1000"/>
</p>

---

<p align="center">
  <img src="images/image (42).png" alt="" width="1000"/>
</p>

---

<p align="center">
  <img src="images/image (43).png" alt="" width="1000"/>
</p>

---
<p align="center">
  <img src="images/image (44).png" alt="" width="1000"/>
</p>

---

## Key Algorithms (Pseudocode)
### Single Neuron Training (MSE)
```python
for epoch in range(E):
    y_hat = X @ W + b          # forward
    loss = mean((y_hat - y)**2)
    dW = (2/N) * X.T @ (y_hat - y)
    db = (2/N) * sum(y_hat - y)
    W -= lr * dW
    b -= lr * db
```
### MLP Backprop Skeleton
```python
# Forward caches per layer
for layer in layers:
    z = a_prev @ W + b
    a = activation(z)
    cache(layer, a_prev, z)
# Backward
delta = 2*(a_L - y)/N
for layer in reversed(layers):
    dz = delta * activation_grad(z)
    grad_W = a_prev.T @ dz
    grad_b = sum(dz)
    delta = dz @ W.T
    update(Adam, W, grad_W)
```
### Coordinate Descent LASSO
```python
for it in range(max_iter):
    for j in range(M):
        residual = y - A @ coef + A[:, j]*coef[j]
        rho = A[:, j] @ residual
        z = rho / (A[:, j] @ A[:, j])
        coef[j] = soft_threshold(z, alpha / (A[:, j] @ A[:, j]))
    if convergence: break
```
### Manual LSTM Cell
```python
def lstm_cell(h_prev, c_prev, x_t):
    gates = linear(concat(h_prev, x_t))  # -> 4*H
    i, f, g, o = split(gates)
    i, f, o = sigmoid(i), sigmoid(f), sigmoid(o)
    g = tanh(g)
    c_t = f * c_prev + i * g
    h_t = o * tanh(c_t)
    return h_t, c_t
```
---
## Experimental Setups
| Experiment | Architecture | Activation Stack | Optimizer | Epochs | Notable Hyperparameters |
|------------|-------------|------------------|-----------|--------|-------------------------|
| Linear Fit | 1→1 | Linear | SGD | 2000 | lr=0.05 |
| MLP (sin) | 1→48→24→1 | tanh,tanh,linear | Adam | 1200 | lr=2e-3, batch=64 |
| MLP (composite) | 1→64→32→1 | tanh,tanh,linear | Adam | 1200 | lr=1e-3 |
| LSTM (composite) | seq(8,3)→LSTM64→MLP(64→1) | tanh in gates | Adam | 80 | lr=1e-3, batch=64 |

(Adjust hyperparameters for noise robustness or faster convergence.)

---
