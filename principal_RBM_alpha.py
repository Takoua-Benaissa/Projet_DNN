"""
Construction d'un RBM et test sur Binary AlphaDigits
Analyses réalisées :
    A1 – Évaluation préliminaire  (entraînement + génération)
    A2 – Influence du nombre d'unités cachées
    A3 – Influence du nombre d'epochs
    A4 – Influence du nombre de classes

"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import lire_alpha_digit
from rbm   import init_RBM, train_RBM, generer_image_RBM

# reproductibilité 
np.random.seed(42)

#  paramètres globaux 
IMAGE_SHAPE = (20, 16)
N_VISIBLE   = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]   # 320
N_GIBBS     = 500          # itérations de Gibbs pour la génération
LEARNING_RATE = 0.1
BATCH_SIZE    = 32

OUT = 'resultats_RBM_alpha'
os.makedirs(OUT, exist_ok=True)




#  Helpers visuels
def plot_images(images, title, filepath, n_cols=10):
    n = len(images)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.7, n_rows * 2.1))
    axes = np.array(axes).reshape(n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            ax  = axes[i, j]
            if idx < n:
                ax.imshow(images[idx].reshape(IMAGE_SHAPE),
                          cmap='gray', vmin=0, vmax=1,
                          interpolation='nearest')
            ax.axis('off')
    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(filepath, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"→ {filepath}")


def plot_loss_curve(history, title, filepath):
    """Courbe MSE de reconstruction par epoch."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(history) + 1), history,
            color='steelblue', linewidth=1.6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE reconstruction')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filepath, dpi=130)
    plt.close()
    print(f" → {filepath}")


def plot_multi_losses(histories, labels, title, filepath):
    fig, ax = plt.subplots(figsize=(7, 4))
    for h, lbl in zip(histories, labels):
        ax.plot(range(1, len(h) + 1), h, linewidth=1.5, label=lbl)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE reconstruction')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filepath, dpi=130)
    plt.close()
    print(f"    → {filepath}")


def plot_bar(x_vals, y_vals, x_label, title, filepath, color='steelblue'):
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([str(v) for v in x_vals], y_vals,
                  color=color, edgecolor='black', alpha=0.85)
    ax.set_xlabel(x_label)
    ax.set_ylabel('MSE reconstruction finale')
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    for bar, v in zip(bars, y_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + max(y_vals) * 0.015,
                f'{v:.5f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=130)
    plt.close()
    print(f"    → {filepath}")




#  A1 – Évaluation préliminaire
print("  A1 – Évaluation préliminaire  (classes A, B | q=500 | 1000 epochs)")

X_AB = lire_alpha_digit(['A', 'B'])
print(f"  Données chargées : {X_AB.shape}  "
      f"({X_AB.shape[0]} images, {X_AB.shape[1]} pixels)")

#  afficher quelques exemples d'entraînement
plot_images(X_AB[:10],
            "Exemples d'entraînement  (classes A et B)",
            os.path.join(OUT, 'A1_train_samples.png'))

#  entraînement
print("\n  Entraînement du RBM…")
rbm_A1 = init_RBM(N_VISIBLE, 500)
rbm_A1, hist_A1 = train_RBM(rbm_A1, X_AB,
                              epochs=1000,
                              learning_rate=LEARNING_RATE,
                              batch_size=BATCH_SIZE,
                              verbose=True)

print(f"\n MSE finale : {hist_A1[-1]:.6f}")

#  courbe d'apprentissage
plot_loss_curve(hist_A1, "A1 – Courbe d'apprentissage  (RBM q=500 | A,B | 1000 epochs)", os.path.join(OUT, 'A1_loss_curve.png'))

#  génération
print(f"\n  Génération de 10 images ({N_GIBBS} itérations de Gibbs)…")
gen_A1 = generer_image_RBM(rbm_A1, N_GIBBS, 10, IMAGE_SHAPE)
plot_images(gen_A1,
            f"A1 – Images générées  (RBM q=500 | A,B | MSE={hist_A1[-1]:.5f})", os.path.join(OUT, 'A1_generated.png'))






#  A2 – Influence du nombre d'unités cachées
print("  A2 – Influence du nombre d'unités cachées  (classes A,B | 1000 epochs)")

hidden_sizes   = [50, 100, 200, 500, 1000]
mse_hidden     = []
histories_hid  = []

for q in hidden_sizes:
    print(f"\n  q = {q}")
    rbm_q = init_RBM(N_VISIBLE, q)
    rbm_q, hist_q = train_RBM(rbm_q, X_AB,
                               epochs=1000,
                               learning_rate=LEARNING_RATE,
                               batch_size=BATCH_SIZE,
                               verbose=False)
    mse_hidden.append(hist_q[-1])
    histories_hid.append(hist_q)
    print(f"    MSE finale = {hist_q[-1]:.6f}")

    gen_q = generer_image_RBM(rbm_q, N_GIBBS, 10, IMAGE_SHAPE)
    plot_images(gen_q,
                f"A2 – Généré  (q={q} | MSE={hist_q[-1]:.5f})",
                os.path.join(OUT, f'A2_generated_q{q}.png'))
#  barplot MSE finale vs q
plot_bar(hidden_sizes, mse_hidden,
         "Nombre d'unités cachées (q)",
         "A2 – MSE finale vs nombre d'unités cachées (RBM)",
         os.path.join(OUT, 'A2_mse_vs_q.png'))

#  courbes d'apprentissage comparées
plot_multi_losses(histories_hid,
                  [f'q={q}' for q in hidden_sizes],
                  "A2 – Courbes d'apprentissage selon q",
                  os.path.join(OUT, 'A2_loss_curves_q.png'))





#  A3 – Influence du nombre d'epochs
print("  A3 – Influence du nombre d'epochs  (classes A,B | q=500)")
epoch_list    = [50, 100, 200, 500, 1000]
mse_epochs    = []
histories_ep  = []

for ep in epoch_list:
    print(f"\n  epochs = {ep}")
    rbm_ep = init_RBM(N_VISIBLE, 500)
    rbm_ep, hist_ep = train_RBM(rbm_ep, X_AB,
                                 epochs=ep,
                                 learning_rate=LEARNING_RATE,
                                 batch_size=BATCH_SIZE,
                                 verbose=False)
    mse_epochs.append(hist_ep[-1])
    histories_ep.append(hist_ep)
    print(f"    MSE finale = {hist_ep[-1]:.6f}")

    gen_ep = generer_image_RBM(rbm_ep, N_GIBBS, 10, IMAGE_SHAPE)
    plot_images(gen_ep,
                f"A3 – Généré  (epochs={ep} | MSE={hist_ep[-1]:.5f})",
                os.path.join(OUT, f'A3_generated_ep{ep}.png'))

#  barplot
plot_bar(epoch_list, mse_epochs,
         "Nombre d'epochs",
         "A3 – MSE finale vs nombre d'epochs (RBM, q=500)",
         os.path.join(OUT, 'A3_mse_vs_epochs.png'),
         color='darkorange')

#  courbes comparées
plot_multi_losses(histories_ep,
                  [f'{e} epochs' for e in epoch_list],
                  "A3 – Courbes d'apprentissage selon le nombre d'epochs",
                  os.path.join(OUT, 'A3_loss_curves_epochs.png'))




#  A4 – Influence du nombre de classes
print("  A4 – Influence du nombre de classes  (q=500 | 1000 epochs)")

class_sets = [
    ['A'],
    ['A', 'B'],
    ['A', 'B', 'C', 'D', 'E'],
]
mse_classes    = []
histories_cls  = []

for cls in class_sets:
    print(f"\n  classes = {cls}  ({len(cls)} classe(s))")
    X_cls = lire_alpha_digit(cls)
    rbm_cls = init_RBM(N_VISIBLE, 500)
    rbm_cls, hist_cls = train_RBM(rbm_cls, X_cls,
                                   epochs=1000,
                                   learning_rate=LEARNING_RATE,
                                   batch_size=BATCH_SIZE,
                                   verbose=False)
    mse_classes.append(hist_cls[-1])
    histories_cls.append(hist_cls)
    print(f"    MSE finale = {hist_cls[-1]:.6f}")

    gen_cls = generer_image_RBM(rbm_cls, N_GIBBS, 10, IMAGE_SHAPE)
    tag = ''.join(cls)
    plot_images(gen_cls,
                f"A4 – Généré  (classes={cls} | MSE={hist_cls[-1]:.5f})",
                os.path.join(OUT, f'A4_generated_{tag}.png'))

#  barplot
plot_bar([len(c) for c in class_sets], mse_classes,
         "Nombre de classes",
         "A4 – MSE finale vs nombre de classes (RBM, q=500)",
         os.path.join(OUT, 'A4_mse_vs_nclasses.png'),
         color='seagreen')

#  courbes comparées
plot_multi_losses(histories_cls,
                  [f'{len(c)} classe(s)' for c in class_sets],
                  "A4 – Courbes d'apprentissage selon le nombre de classes",
                  os.path.join(OUT, 'A4_loss_curves_classes.png'))

#  Récapitulatif
print("  Récapitulatif des MSE finales – RBM")
print("\n  A2  –  Influence du nombre d'unités cachées (classes A,B)")
for q, m in zip(hidden_sizes, mse_hidden):
    print(f"    q = {q:5d}  →  MSE = {m:.6f}")
print("\n  A3  –  Influence du nombre d'epochs (q=500, classes A,B)")
for e, m in zip(epoch_list, mse_epochs):
    print(f"    epochs = {e:5d}  →  MSE = {m:.6f}")
print("\n  A4  –  Influence du nombre de classes (q=500, 1000 epochs)")
for cls, m in zip(class_sets, mse_classes):
    print(f"    {len(cls):2d} classe(s)  →  MSE = {m:.6f}")
print(f"\nTous les résultats ont été sauvegardés dans '{OUT}/'")
print("Terminé.\n")