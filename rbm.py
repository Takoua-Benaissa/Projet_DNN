import numpy as np
import torch
import torch.nn as nn


#  device 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RBM:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        self.W = None   # (n_visible, n_hidden)
        self.a = None   # (1, n_visible)
        self.b = None   # (1, n_hidden)


#  helpers 
def _to_tensor(x):
    """Convert numpy array → float32 tensor on DEVICE (no-op if already tensor)."""
    if isinstance(x, torch.Tensor):
        return x.to(DEVICE, dtype=torch.float32)
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


def _sigmoid(x):
    return torch.sigmoid(x)


#  construction 
def init_RBM(n_visible, n_hidden):
    rbm   = RBM(n_visible, n_hidden)
    rbm.W = torch.randn(n_visible, n_hidden, device=DEVICE) * 0.01
    rbm.a = torch.zeros(1, n_visible, device=DEVICE)
    rbm.b = torch.zeros(1, n_hidden,  device=DEVICE)
    return rbm


#  forward / backward 
def entree_sortie_RBM(rbm, V):
    V = _to_tensor(V)
    return _sigmoid(V @ rbm.W + rbm.b)


def sortie_entree_RBM(rbm, H):
    H = _to_tensor(H)
    return _sigmoid(H @ rbm.W.t() + rbm.a)


#  training 
def train_RBM(rbm, X, epochs=100, learning_rate=0.1,
              batch_size=128, verbose=True):
    X = _to_tensor(X)
    n_samples = X.shape[0]
    history   = []

    for epoch in range(epochs):
        idx    = torch.randperm(n_samples, device=DEVICE)
        X_shuf = X[idx]
        total_err = 0.0

        for start in range(0, n_samples, batch_size):
            V0 = X_shuf[start: start + batch_size]
            mb = V0.shape[0]

            # positive phase
            H0_prob   = entree_sortie_RBM(rbm, V0)
            H0_sample = (H0_prob > torch.rand_like(H0_prob)).float()

            # negative phase (CD-1)
            V1_prob  = sortie_entree_RBM(rbm, H0_sample)
            H1_prob  = entree_sortie_RBM(rbm, V1_prob)

            # gradients
            dW = (V0.t() @ H0_prob - V1_prob.t() @ H1_prob) / mb
            da = (V0 - V1_prob).mean(dim=0, keepdim=True)
            db = (H0_prob - H1_prob).mean(dim=0, keepdim=True)

            rbm.W += learning_rate * dW
            rbm.a += learning_rate * da
            rbm.b += learning_rate * db

            total_err += ((V0 - V1_prob) ** 2).sum().item()

        mse = total_err / n_samples
        history.append(mse)
        if verbose:
            print(f"  Epoch {epoch+1:4d}/{epochs}  –  MSE: {mse:.6f}")

    return rbm, history


#  generation 
def generer_image_RBM(rbm, n_iterations, n_images, image_shape=(20, 16)):
    V = (torch.rand(n_images, rbm.n_visible, device=DEVICE) > 0.5).float()

    for _ in range(n_iterations):
        H_prob   = entree_sortie_RBM(rbm, V)
        H_sample = (H_prob > torch.rand_like(H_prob)).float()
        V_prob   = sortie_entree_RBM(rbm, H_sample)
        V        = (V_prob > torch.rand_like(V_prob)).float()

    return V_prob.cpu().numpy()