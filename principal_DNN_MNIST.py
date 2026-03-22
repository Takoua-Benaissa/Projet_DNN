"""
principal_DNN_MNIST.py  –  GPU-accelerated (PyTorch + CUDA)
=============================================================
Étude comparative DNN pré-entraîné vs aléatoire sur MNIST.

Figures :
    Fig 1 – Taux d'erreur vs nombre de couches
    Fig 2 – Taux d'erreur vs nombre de neurones par couche
    Fig 3 – Taux d'erreur vs taille du jeu d'entraînement
    + Préliminaire + Meilleure configuration
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
print(f"PyTorch {torch.__version__}  –  GPU disponible : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device : {torch.cuda.get_device_name(0)}")

from utils import load_mnist
from dnn   import (init_DNN, pretrain_DNN, retropropagation,
                   test_DNN, entree_sortie_reseau, DEVICE)

# ── reproductibilité ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── dossier de résultats ──────────────────────────────────────────────────────
OUT = 'resultats_DNN_MNIST'
os.makedirs(OUT, exist_ok=True)

# ── hyperparamètres ───────────────────────────────────────────────────────────
N_VISIBLE  = 784
N_CLASSES  = 10
LR         = 0.1
BATCH_SIZE = 128
EPOCHS_RBM = 100
EPOCHS_BP  = 200


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def train_and_eval(X_tr, Y_tr, X_te, Y_te, network_shape,
                   pretrain=True, tag=''):
    dnn = init_DNN(network_shape)

    if pretrain:
        print(f"    [{tag}] Pré-entraînement DBN…")
        dnn, _ = pretrain_DNN(dnn, X_tr,
                              epochs=EPOCHS_RBM, learning_rate=LR,
                              batch_size=BATCH_SIZE, verbose=False)

    print(f"    [{tag}] Rétropropagation…")
    dnn, hist_bp = retropropagation(dnn, X_tr, Y_tr,
                                    epochs=EPOCHS_BP, learning_rate=LR,
                                    batch_size=BATCH_SIZE, verbose=False)

    err_tr = test_DNN(dnn, X_tr, Y_tr)
    err_te = test_DNN(dnn, X_te, Y_te)
    print(f"    [{tag}] Err train={err_tr*100:.2f}%  Err test={err_te*100:.2f}%")
    return err_tr, err_te, hist_bp, dnn


def plot_comparison(x_vals, err_pre_tr, err_pre_te,
                    err_rnd_tr, err_rnd_te,
                    x_label, title, filepath):
    """
    Produit 2 figures séparées :
      - _train.png : 2 plots côte à côte  (pré-entraîné TRAIN | aléatoire TRAIN)
      - _test.png  : 2 plots côte à côte  (pré-entraîné TEST  | aléatoire TEST)
    Chaque plot contient une seule courbe.
    """
    x_str = [str(v) for v in x_vals]

    for split, err_pre, err_rnd in [
            ('train', err_pre_tr, err_rnd_tr),
            ('test',  err_pre_te, err_rnd_te)]:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

        # ── plot gauche : réseau pré-entraîné ─────────────────────────────────
        ax1.plot(x_str, [e * 100 for e in err_pre],
                 'o-', color='steelblue', lw=2, ms=7)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("Taux d'erreur (%)")
        ax1.set_title(f'Réseau pré-entraîné  ') #[{split.upper()}]
        ax1.grid(True, linestyle='--', alpha=0.5)
        # annoter chaque point
        for x, y in zip(x_str, [e * 100 for e in err_pre]):
            ax1.annotate(f'{y:.2f}%', (x, y),
                         textcoords='offset points', xytext=(0, 8),
                         ha='center', fontsize=8, color='steelblue')

        # ── plot droit : réseau aléatoire ─────────────────────────────────────
        ax2.plot(x_str, [e * 100 for e in err_rnd],
                 's--', color='darkorange', lw=2, ms=7)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel("Taux d'erreur (%)")
        ax2.set_title(f'Réseau aléatoire  ') #[{split.upper()}]
        ax2.grid(True, linestyle='--', alpha=0.5)
        for x, y in zip(x_str, [e * 100 for e in err_rnd]):
            ax2.annotate(f'{y:.2f}%', (x, y),
                         textcoords='offset points', xytext=(0, 8),
                         ha='center', fontsize=8, color='darkorange')

        fig.suptitle(f'{title}  –  {split.upper()}', fontsize=12, y=1.02)
        plt.tight_layout()

        # filepath_base = ex: 'resultats/fig1_error_vs_nlayers'
        filepath_base = filepath.replace('.png', '')
        out = f'{filepath_base}_{split}.png'
        plt.savefig(out, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  → {out}")


def plot_loss_curve(hist_pre, hist_rnd, title, filepath):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(hist_pre) + 1), hist_pre,
            color='steelblue',  lw=1.8, label='Pré-entraîné')
    ax.plot(range(1, len(hist_rnd) + 1), hist_rnd,
            color='darkorange', lw=1.8, ls='--', label='Aléatoire')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-Entropy')
    ax.set_title(title); ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filepath, dpi=130)
    plt.close()
    print(f"  → {filepath}")


def show_output_probs(dnn, X, Y, n=5, filepath=None):
    outputs = entree_sortie_reseau(dnn, X[:n])
    probs   = outputs[-1].cpu().numpy()
    y_true  = np.argmax(Y[:n], axis=1)
    fig, axes = plt.subplots(n, 2, figsize=(9, n * 2.0))
    for i in range(n):
        axes[i, 0].imshow(X[i].reshape(28, 28), cmap='gray')
        axes[i, 0].set_title(f'Vraie classe : {y_true[i]}', fontsize=9)
        axes[i, 0].axis('off')
        axes[i, 1].bar(range(10), probs[i], color='steelblue',
                       edgecolor='black', alpha=0.8)
        axes[i, 1].set_xticks(range(10))
        axes[i, 1].set_title(
            f'Prédit : {np.argmax(probs[i])}  (conf={probs[i].max()*100:.1f}%)',
            fontsize=9)
        axes[i, 1].set_ylim(0, 1)
        axes[i, 1].grid(True, axis='y', ls='--', alpha=0.4)
    plt.suptitle('Probabilités de sortie – DNN pré-entraîné', fontsize=11)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  → {filepath}")


# ═════════════════════════════════════════════════════════════════════════════
#  Chargement MNIST
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Chargement MNIST")
print("=" * 65)

X_train, Y_train, X_test, Y_test = load_mnist()
print(f"  Train : {X_train.shape}   Test : {X_test.shape}")


# ═════════════════════════════════════════════════════════════════════════════
#  Préliminaire  [784, 200, 200, 10]
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Préliminaire  –  [784, 200, 200, 10]")
print("=" * 65)

ARCH_REF = [N_VISIBLE, 200, 200, N_CLASSES]

err_pre_tr0, err_pre_te0, hist_pre0, dnn_pre0 = train_and_eval(
    X_train, Y_train, X_test, Y_test, ARCH_REF,
    pretrain=True, tag='pré-entraîné')

err_rnd_tr0, err_rnd_te0, hist_rnd0, _ = train_and_eval(
    X_train, Y_train, X_test, Y_test, ARCH_REF,
    pretrain=False, tag='aléatoire')

plot_loss_curve(hist_pre0, hist_rnd0,
                "Préliminaire – Cross-Entropy  [784,200,200,10]",
                os.path.join(OUT, 'prelim_loss_curves.png'))

show_output_probs(dnn_pre0, X_train, Y_train, n=5,
                  filepath=os.path.join(OUT, 'prelim_output_probs.png'))

print(f"\n  Pré-entraîné : train={err_pre_tr0*100:.2f}%  test={err_pre_te0*100:.2f}%")
print(f"  Aléatoire    : train={err_rnd_tr0*100:.2f}%  test={err_rnd_te0*100:.2f}%")


# ═════════════════════════════════════════════════════════════════════════════
#  Fig 1 – vs nombre de couches  (200 neurones / couche)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Fig 1  –  Taux d'erreur vs nombre de couches cachées")
print("=" * 65)

n_layers_list = [1, 2, 3, 4, 5]
err_pre_tr_L, err_pre_te_L = [], []
err_rnd_tr_L, err_rnd_te_L = [], []

for nl in n_layers_list:
    shape = [N_VISIBLE] + [200] * nl + [N_CLASSES]
    print(f"\n  {nl} couche(s)  →  {shape}")
    etr_p, ete_p, _, _ = train_and_eval(
        X_train, Y_train, X_test, Y_test, shape,
        pretrain=True,  tag=f'pré nl={nl}')
    etr_r, ete_r, _, _ = train_and_eval(
        X_train, Y_train, X_test, Y_test, shape,
        pretrain=False, tag=f'rnd nl={nl}')
    err_pre_tr_L.append(etr_p); err_pre_te_L.append(ete_p)
    err_rnd_tr_L.append(etr_r); err_rnd_te_L.append(ete_r)

plot_comparison(n_layers_list,
                err_pre_tr_L, err_pre_te_L,
                err_rnd_tr_L, err_rnd_te_L,
                x_label="Nombre de couches cachées",
                title="Fig 1 – Taux d'erreur vs nombre de couches",
                filepath=os.path.join(OUT, 'fig1_error_vs_nlayers.png'))


# ═════════════════════════════════════════════════════════════════════════════
#  Fig 2 – vs nombre de neurones  (2 couches cachées)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Fig 2  –  Taux d'erreur vs nombre de neurones/couche")
print("=" * 65)

neuron_list = [100, 200, 300, 500, 700]
err_pre_tr_N, err_pre_te_N = [], []
err_rnd_tr_N, err_rnd_te_N = [], []

for q in neuron_list:
    shape = [N_VISIBLE, q, q, N_CLASSES]
    print(f"\n  q = {q}  →  {shape}")
    etr_p, ete_p, _, _ = train_and_eval(
        X_train, Y_train, X_test, Y_test, shape,
        pretrain=True,  tag=f'pré q={q}')
    etr_r, ete_r, _, _ = train_and_eval(
        X_train, Y_train, X_test, Y_test, shape,
        pretrain=False, tag=f'rnd q={q}')
    err_pre_tr_N.append(etr_p); err_pre_te_N.append(ete_p)
    err_rnd_tr_N.append(etr_r); err_rnd_te_N.append(ete_r)

plot_comparison(neuron_list,
                err_pre_tr_N, err_pre_te_N,
                err_rnd_tr_N, err_rnd_te_N,
                x_label="Nombre de neurones par couche",
                title="Fig 2 – Taux d'erreur vs neurones/couche",
                filepath=os.path.join(OUT, 'fig2_error_vs_neurons.png'))


# ═════════════════════════════════════════════════════════════════════════════
#  Fig 3 – vs taille du jeu d'entraînement  ([784, 200, 200, 10])
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Fig 3  –  Taux d'erreur vs taille du jeu d'entraînement")
print("=" * 65)

ARCH_FIG3    = [N_VISIBLE, 200, 200, N_CLASSES]
sample_sizes = [1000, 3000, 7000, 10000, 30000, 60000]
err_pre_tr_S, err_pre_te_S = [], []
err_rnd_tr_S, err_rnd_te_S = [], []

for n_tr in sample_sizes:
    print(f"\n  n_train = {n_tr}")
    etr_p, ete_p, _, _ = train_and_eval(
        X_train[:n_tr], Y_train[:n_tr], X_test, Y_test, ARCH_FIG3,
        pretrain=True,  tag=f'pré n={n_tr}')
    etr_r, ete_r, _, _ = train_and_eval(
        X_train[:n_tr], Y_train[:n_tr], X_test, Y_test, ARCH_FIG3,
        pretrain=False, tag=f'rnd n={n_tr}')
    err_pre_tr_S.append(etr_p); err_pre_te_S.append(ete_p)
    err_rnd_tr_S.append(etr_r); err_rnd_te_S.append(ete_r)

plot_comparison(sample_sizes,
                err_pre_tr_S, err_pre_te_S,
                err_rnd_tr_S, err_rnd_te_S,
                x_label="Nombre de données d'entraînement",
                title="Fig 3 – Taux d'erreur vs taille du jeu d'entraînement",
                filepath=os.path.join(OUT, 'fig3_error_vs_nsamples.png'))


# ═════════════════════════════════════════════════════════════════════════════
#  Meilleure configuration
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Meilleure configuration")
print("=" * 65)

best_candidates = [
    [N_VISIBLE, 300, 300, N_CLASSES],
    [N_VISIBLE, 500, 500, N_CLASSES],
    [N_VISIBLE, 300, 300, 300, N_CLASSES],
    [N_VISIBLE, 500, 500, 500, N_CLASSES],
]
best_err, best_arch, best_hist = 1.0, None, None

for shape in best_candidates:
    print(f"\n  Candidat : {shape}")
    etr, ete, hist_bp, _ = train_and_eval(
        X_train, Y_train, X_test, Y_test, shape,
        pretrain=True, tag='best')
    if ete < best_err:
        best_err  = ete
        best_arch = shape
        best_hist = hist_bp

print(f"\n  ★  Meilleure architecture : {best_arch}")
print(f"  ★  Taux d'erreur test     : {best_err*100:.2f}%")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, len(best_hist)+1), best_hist,
        color='steelblue', lw=1.8)
ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-Entropy')
ax.set_title(f'Meilleure config {best_arch}')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'best_config_loss.png'), dpi=130)
plt.close()


# ═════════════════════════════════════════════════════════════════════════════
#  Récapitulatif
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Récapitulatif")
print("=" * 65)
print(f"\n  Préliminaire [784,200,200,10]")
print(f"    Pré  : train={err_pre_tr0*100:.2f}%  test={err_pre_te0*100:.2f}%")
print(f"    Rnd  : train={err_rnd_tr0*100:.2f}%  test={err_rnd_te0*100:.2f}%")
print("\n  Fig 1  (vs couches)")
for nl, ep, er in zip(n_layers_list, err_pre_te_L, err_rnd_te_L):
    print(f"    {nl} couche(s) : pré={ep*100:.2f}%  rnd={er*100:.2f}%  [test]")
print("\n  Fig 2  (vs neurones)")
for q, ep, er in zip(neuron_list, err_pre_te_N, err_rnd_te_N):
    print(f"    q={q:4d} : pré={ep*100:.2f}%  rnd={er*100:.2f}%  [test]")
print("\n  Fig 3  (vs n_train)")
for n, ep, er in zip(sample_sizes, err_pre_te_S, err_rnd_te_S):
    print(f"    n={n:6d} : pré={ep*100:.2f}%  rnd={er*100:.2f}%  [test]")
print(f"\n  ★  Meilleure architecture : {best_arch}")
print(f"  ★  Taux d'erreur test     : {best_err*100:.2f}%")
print(f"\nRésultats sauvegardés dans '{OUT}/'")
print("Terminé.\n")