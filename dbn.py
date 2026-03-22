import torch
from rbm import init_RBM, train_RBM, entree_sortie_RBM, sortie_entree_RBM, DEVICE


def init_DBN(network_shape):
    dbn = []
    for i in range(len(network_shape) - 1):
        dbn.append(init_RBM(network_shape[i], network_shape[i + 1]))
    return dbn


def train_DBN(dbn, X, epochs=100, learning_rate=0.1, batch_size=128, verbose=True):

    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    current_input = X.clone()
    histories = []

    for idx, rbm in enumerate(dbn):
        if verbose:
            print(f"\n=== DBN layer {idx+1}/{len(dbn)}"
                  f"  ({rbm.n_visible} → {rbm.n_hidden}) ===")

        rbm, history = train_RBM(
            rbm, current_input,
            epochs=epochs, learning_rate=learning_rate,
            batch_size=batch_size, verbose=verbose
        )
        histories.append(history)
        current_input = entree_sortie_RBM(rbm, current_input).detach()

    return dbn, histories


def generer_image_DBN(dbn, n_iterations, n_images, image_shape=(20, 16)):
    top_rbm = dbn[-1]
    V_top = (torch.rand(n_images, top_rbm.n_visible, device=DEVICE) > 0.5).float()

    for _ in range(n_iterations):
        H_prob     = entree_sortie_RBM(top_rbm, V_top)
        H_sample   = (H_prob > torch.rand_like(H_prob)).float()
        V_top_prob = sortie_entree_RBM(top_rbm, H_sample)
        V_top      = (V_top_prob > torch.rand_like(V_top_prob)).float()

    current = V_top_prob
    for layer_idx in range(len(dbn) - 2, -1, -1):
        current = sortie_entree_RBM(dbn[layer_idx], current)

    return current.cpu().numpy()