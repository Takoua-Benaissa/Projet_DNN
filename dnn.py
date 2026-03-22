
import numpy as np
import torch
import torch.nn.functional as F
from rbm import RBM, init_RBM, entree_sortie_RBM, DEVICE
from dbn import init_DBN, train_DBN


#  helpers 
def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(DEVICE, dtype=torch.float32)
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


#  construction 
def init_DNN(network_shape):
    return init_DBN(network_shape)

#  pre-training 
def pretrain_DNN(dnn, X, epochs=100, learning_rate=0.1,
                 batch_size=128, verbose=True):
    hidden_dbn = dnn[:-1]
    hidden_dbn, histories = train_DBN(
        hidden_dbn, X,
        epochs=epochs, learning_rate=learning_rate,
        batch_size=batch_size, verbose=verbose
    )
    for i, rbm in enumerate(hidden_dbn):
        dnn[i] = rbm
    return dnn, histories


#  forward passes 

def calcul_softmax(rbm, A):
    A = _to_tensor(A)
    logits = A @ rbm.W + rbm.b
    return F.softmax(logits, dim=1)


def entree_sortie_reseau(dnn, X):
    X = _to_tensor(X)
    layer_outputs = [X]
    current = X
    for rbm in dnn[:-1]:
        current = torch.sigmoid(current @ rbm.W + rbm.b)
        layer_outputs.append(current)
    probs = calcul_softmax(dnn[-1], current)
    layer_outputs.append(probs)
    return layer_outputs


#  supervised fine-tuning 

def retropropagation(dnn, X, Y, epochs=200, learning_rate=0.1,
                     batch_size=128, verbose=True):
    X = _to_tensor(X)
    Y = _to_tensor(Y)
    n_samples = X.shape[0]
    n_layers  = len(dnn)
    history   = []

    for epoch in range(epochs):
        idx    = torch.randperm(n_samples, device=DEVICE)
        X_shuf = X[idx]
        Y_shuf = Y[idx]
        total_loss = 0.0

        for start in range(0, n_samples, batch_size):
            Xb = X_shuf[start: start + batch_size]
            Yb = Y_shuf[start: start + batch_size]
            mb = Xb.shape[0]

            #  forward ─
            outputs = entree_sortie_reseau(dnn, Xb)
            probs   = outputs[-1]

            # cross-entropy
            loss = -(Yb * torch.log(probs + 1e-12)).sum(dim=1).mean()
            total_loss += loss.item() * mb

            #  backward 
            deltas = [None] * n_layers
            deltas[-1] = probs - Yb

            for l in range(n_layers - 2, -1, -1):
                a = outputs[l + 1]
                deltas[l] = (deltas[l + 1] @ dnn[l + 1].W.t()) * a * (1.0 - a)

            #  parameter updates ─
            for l in range(n_layers):
                a_in = outputs[l]
                dW   = a_in.t() @ deltas[l] / mb
                db   = deltas[l].mean(dim=0, keepdim=True)

                dnn[l].W -= learning_rate * dW
                dnn[l].b -= learning_rate * db
                if l > 0:
                    da = (deltas[l].mean(dim=0, keepdim=True) @ dnn[l].W.t())
                    dnn[l].a -= learning_rate * da

        mean_loss = total_loss / n_samples
        history.append(mean_loss)
        if verbose:
            print(f"  Epoch {epoch+1:4d}/{epochs}  –  Cross-Entropy: {mean_loss:.6f}")

    return dnn, history


#  evaluation 

def test_DNN(dnn, X, Y):
    X = _to_tensor(X)
    Y = _to_tensor(Y)
    outputs = entree_sortie_reseau(dnn, X)
    probs   = outputs[-1]
    y_pred  = torch.argmax(probs, dim=1)
    y_true  = torch.argmax(Y,    dim=1)
    return (y_pred != y_true).float().mean().item()