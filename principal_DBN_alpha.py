"""
Analyses réalisées :
    B1 – Évaluation préliminaire  (entraînement + génération)
    B2 – Influence du nombre d'unités cachées
    B3 – Influence du nombre de couches (RBMs empilés)
    B4 – Influence du nombre de classes

"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import lire_alpha_digit
from dbn   import init_DBN, train_DBN, generer_image_DBN

#  reproductibilité 
np.random.seed(42)

#  paramètres globaux 
IMAGE_SHAPE   = (20, 16)
N_VISIBLE     = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]   # 320
N_GIBBS       = 500
LEARNING_RATE = 0.1
BATCH_SIZE    = 32
EPOCHS        = 1000

OUT = 'resultats_DBN_alpha'
os.makedirs(OUT, exist_ok=True)



#  Helpers visuels
def plot_images(images, title, filepath, n_cols=10):
    n      = len(images)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.7, n_rows * 2.1))
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
    print(f"    → {filepath}")


def plot_loss_curves_dbn(histories, architecture, title, filepath):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, h in enumerate(histories):
        ax.plot(range(1, len(h) + 1), h, linewidth=1.5,
                label=f'Couche {i+1}  '
                      f'({architecture[i]}→{architecture[i+1]})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE reconstruction')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filepath, dpi=130)
    plt.close()
    print(f"    → {filepath}")


def plot_multi_losses(histories, labels, title, filepath):
    """Plusieurs courbes MSE (couche 1) pour comparer les configs."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for h, lbl in zip(histories, labels):
        ax.plot(range(1, len(h) + 1), h, linewidth=1.5, label=lbl)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE reconstruction  (couche 1)')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filepath, dpi=130)
    plt.close()
    print(f"    → {filepath}")


def plot_bar(x_vals, y_vals, x_label, title, filepath, color='steelblue'):
    """Barplot MSE finale vs un hyperparamètre."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([str(v) for v in x_vals], y_vals,
                  color=color, edgecolor='black', alpha=0.85)
    ax.set_xlabel(x_label)
    ax.set_ylabel('MSE reconstruction finale  (couche 1)')
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

#  B1 – Évaluation préliminaire
print("  B1 – Évaluation préliminaire  "
      "(architecture [320,200,100] | classes A,B | 1000 epochs)")

X_AB = lire_alpha_digit(['A', 'B'])
print(f"  Données chargées : {X_AB.shape}")
#  afficher quelques exemples d'entraînement
plot_images(X_AB[:10],
            "Exemples d'entraînement  (classes A et B)", os.path.join(OUT, 'B1_train_samples.png'))

#  architecture préliminaire
arch_B1 = [N_VISIBLE, 200, 100]
print(f"\n  Architecture : {arch_B1}")
print("  Entraînement du DBN…")
dbn_B1, hists_B1 = train_DBN(
    init_DBN(arch_B1), X_AB,
    epochs=EPOCHS, 
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE, 
    verbose=True
)

#  courbes d'apprentissage par couche
plot_loss_curves_dbn(hists_B1, arch_B1,
                     "B1 – Courbes d'apprentissage par couche  "
                     "(DBN [320,200,100] | A,B)",
                     os.path.join(OUT, 'B1_loss_curves.png'))

#  génération
print(f"\n  Génération de 10 images ({N_GIBBS} itérations de Gibbs)…")
gen_B1 = generer_image_DBN(dbn_B1, N_GIBBS, 10, IMAGE_SHAPE)
plot_images(gen_B1,
            f"B1 – Images générées  "
            f"(DBN {arch_B1} | MSE(c1)={hists_B1[0][-1]:.5f})",
            os.path.join(OUT, 'B1_generated.png'))

print(f"\n  MSE couche 1 finale : {hists_B1[0][-1]:.6f}")
print(f"  MSE couche 2 finale : {hists_B1[1][-1]:.6f}")



#  B2 – Influence du nombre d'unités cachées
print("  B2 – Influence du nombre d'unités cachées  "
      "(2 couches | classes A,B | 1000 epochs)")

hidden_sizes  = [50, 100, 200, 500,1000]
mse_hidden    = []
hist_hid_c1   = []   # courbes couche 1 pour comparaison

for q in hidden_sizes:
    arch = [N_VISIBLE, q, q]
    print(f"\n  q = {q}  →  architecture {arch}")
    dbn_q, hists_q = train_DBN(
        init_DBN(arch), X_AB,
        epochs=EPOCHS, learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE, verbose=False
    )
    mse = hists_q[0][-1]
    mse_hidden.append(mse)
    hist_hid_c1.append(hists_q[0])
    print(f"MSE couche 1 finale = {mse:.6f}")

    gen_q = generer_image_DBN(dbn_q, N_GIBBS, 10, IMAGE_SHAPE)
    plot_images(gen_q,
                f"B2 – Généré  (q={q} | MSE(c1)={mse:.5f})",
                os.path.join(OUT, f'B2_generated_q{q}.png'))
#  barplot
plot_bar(hidden_sizes, mse_hidden,
         "Nombre d'unités cachées par couche (q)",
         "B2 – MSE finale (couche 1) vs nombre d'unités cachées (DBN)",
         os.path.join(OUT, 'B2_mse_vs_q.png'))

#  courbes comparées (couche 1)
plot_multi_losses(hist_hid_c1,
                  [f'q={q}' for q in hidden_sizes],
                  "B2 – Courbes d'apprentissage couche 1 selon q",
                  os.path.join(OUT, 'B2_loss_curves_q.png'))


#  B3 – Influence du nombre de couches
print("  B3 – Influence du nombre de couches  "
      "(q=500 par couche | classes A,B | 1000 epochs)")
n_layers_list = [1, 2, 3, 4, 5]
mse_layers    = []
hist_lay_c1   = []
for nl in n_layers_list:
    arch = [N_VISIBLE] + [500] * nl
    print(f"\n  {nl} couche(s)  →  architecture {arch}")
    dbn_nl, hists_nl = train_DBN(
        init_DBN(arch), X_AB,
        epochs=EPOCHS, learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE, verbose=False
    )
    mse = hists_nl[0][-1]
    mse_layers.append(mse)
    hist_lay_c1.append(hists_nl[0])
    print(f"    MSE couche 1 finale = {mse:.6f}")

    gen_nl = generer_image_DBN(dbn_nl, N_GIBBS, 10, IMAGE_SHAPE)
    plot_images(gen_nl,
                f"B3 – Généré  ({nl} couche(s) | MSE(c1)={mse:.5f})",
                os.path.join(OUT, f'B3_generated_nl{nl}.png'))

#  barplot
plot_bar(n_layers_list, mse_layers,
         "Nombre de couches cachées",
         "B3 – MSE finale (couche 1) vs nombre de couches (DBN, q=500)",
         os.path.join(OUT, 'B3_mse_vs_nlayers.png'),
         color='darkorange')
#  courbes comparées (couche 1)
plot_multi_losses(hist_lay_c1,[f'{nl} couche(s)' for nl in n_layers_list],
                  "B3 – Courbes d'apprentissage couche 1 selon le nombre de couches", 
                  os.path.join(OUT, 'B3_loss_curves_nlayers.png'))



#  B4 – Influence du nombre de classes
print("  B4 – Influence du nombre de classes  "
"(architecture [320,200,100] | 1000 epochs)")

class_sets = [
    ['A'],
    ['A', 'B'],
    ['A', 'B', 'C', 'D', 'E'],
]
mse_classes   = []
hist_cls_c1   = []

for cls in class_sets:
    print(f"\n  classes = {cls}  ({len(cls)} classe(s))")
    X_cls = lire_alpha_digit(cls)
    dbn_cls, hists_cls = train_DBN(
        init_DBN([N_VISIBLE, 200, 100]), X_cls,
        epochs=EPOCHS, learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE, verbose=False
    )
    mse = hists_cls[0][-1]
    mse_classes.append(mse)
    hist_cls_c1.append(hists_cls[0])
    print(f"    MSE couche 1 finale = {mse:.6f}")

    gen_cls = generer_image_DBN(dbn_cls, N_GIBBS, 10, IMAGE_SHAPE)
    tag = ''.join(cls)
    plot_images(gen_cls,
                f"B4 – Généré  (classes={cls} | MSE(c1)={mse:.5f})",
                os.path.join(OUT, f'B4_generated_{tag}.png'))

#  barplot
plot_bar([len(c) for c in class_sets], mse_classes,
         "Nombre de classes",
         "B4 – MSE finale (couche 1) vs nombre de classes (DBN)",
         os.path.join(OUT, 'B4_mse_vs_nclasses.png'),
         color='seagreen')

#  courbes comparées (couche 1)
plot_multi_losses(hist_cls_c1,
                  [f'{len(c)} classe(s)' for c in class_sets],
                  "B4 – Courbes d'apprentissage couche 1 selon le nombre de classes",
                  os.path.join(OUT, 'B4_loss_curves_classes.png'))
#  Récapitulatif
print("  Récapitulatif des MSE finales – DBN (couche 1)")
print("\n  B2  –  Influence du nombre d'unités cachées (2 couches, classes A,B)")
for q, m in zip(hidden_sizes, mse_hidden):
    print(f"    q = {q:5d}  →  MSE = {m:.6f}")

print("\n  B3  –  Influence du nombre de couches (q=500, classes A,B 1000 epochs)")
for nl, m in zip(n_layers_list, mse_layers):
    print(f"{nl} couche(s)  →  MSE = {m:.6f}")

print("\n  B4  –  Influence du nombre de classes (arch [320,200,100] | 1000 epochs)")
for cls, m in zip(class_sets, mse_classes):
    print(f"    {len(cls):2d} classe(s)  →  MSE = {m:.6f}")

print(f"\nTous les résultats ont été sauvegardés dans '{OUT}/'")
print("Terminé.\n")