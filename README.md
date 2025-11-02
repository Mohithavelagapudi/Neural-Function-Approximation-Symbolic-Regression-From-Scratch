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
### 1. Linear Neuron (Single Layer)
Forward: \( \hat{y} = x w + b \).  
Loss (MSE): \( \mathcal{L} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2 \).  
Gradients: \( \partial_w = \frac{2}{N} \sum x_i (x_i w + b - y_i), \quad \partial_b = \frac{2}{N} \sum (x_i w + b - y_i) \).  
Parameter update (learning rate \(\eta\)): \( w \leftarrow w - \eta \partial_w, \ b \leftarrow b - \eta \partial_b \).

### 2. Multilayer Perceptron (MLP)
Layer \( \ell \): pre-activation \( z^{(\ell)} = a^{(\ell-1)} W^{(\ell)} + b^{(\ell)} \).  
Activation (tanh/ReLU/linear): \( a^{(\ell)} = \phi(z^{(\ell)}) \).  
Output MSE loss identical form.  
Backward (chain rule): For output gradient \( \delta^{(L)} = \frac{2}{N}(a^{(L)} - y) \). Hidden: \( \delta^{(\ell)} = (\delta^{(\ell+1)} W^{(\ell+1)T}) \odot \phi'(z^{(\ell)}) \).  
Parameter gradients: \( \nabla_{W^{(\ell)}} = a^{(\ell-1)T} \delta^{(\ell)}, \ \nabla_{b^{(\ell)}} = \sum \delta^{(\ell)} \).

### 3. Adam Optimizer (Per-parameter):
Moments: \( m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \), \( v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \).  
Bias correction: \( \hat{m}_t = m_t/(1-\beta_1^t), \hat{v}_t = v_t/(1-\beta_2^t) \).  
Update: \( \theta_t = \theta_{t-1} - \eta \hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon) \).

### 4. LASSO Sparse Symbolic Regression
Objective: \( \min_{\alpha} \frac{1}{2}\|A \alpha - y\|_2^2 + \lambda \|\alpha\|_1 \).  
Coordinate descent update (feature j): soft-threshold on \( z_j = \rho_j / \|A_j\|_2^2 \) where \( \rho_j = A_j^T (y - A \alpha + A_j \alpha_j) \).

### 5. Manual LSTM Cell
Concatenate hidden & input: \( h_{t-1} \Vert x_t \xrightarrow{} [i_t, f_t, g_t, o_t] \).  
Gates: \( i_t = \sigma(W_i [h_{t-1}, x_t]), f_t = \sigma(W_f [...]), o_t = \sigma(W_o [...]), g_t = \tanh(W_g [...]) \).  
State: \( c_t = f_t \odot c_{t-1} + i_t \odot g_t \), \( h_t = o_t \odot \tanh(c_t) \).  
Regression head: \( y = \text{MLP}(h_T) \).

### 6. Basis Engineering & Symbolic Distillation
Basis matrix A includes columns: \(1, x, x^2, ..., x^d, \sin(x), \cos(x), e^{x}, e^{x/2}, ...\). Coefficients -> human-readable expression by thresholding small magnitudes.

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
