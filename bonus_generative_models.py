"""
Comparaison de 4 modèles génératifs sur MNIST (chiffres 5 et 8) :
    1. RBM   2. DBN   3. VAE   4. GAN
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

print(f"PyTorch {torch.__version__}  –  GPU: {torch.cuda.is_available()}")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils import load_mnist
from rbm   import init_RBM, train_RBM, generer_image_RBM
from dbn   import init_DBN, train_DBN, generer_image_DBN

torch.manual_seed(42)
np.random.seed(42)

OUT = 'resultats_bonus'
os.makedirs(OUT, exist_ok=True)

N_VISIBLE  = 784
LATENT_DIM = 64
HIDDEN     = 256
N_EPOCHS_RBM = 50
N_EPOCHS_DL  = 100
BATCH_SIZE   = 256         
LR           = 1e-3
N_GIBBS      = 500
N_GEN        = 10



#  Helpers


def plot_generated(images, model_name, filepath, n_cols=10,
                   image_shape=(28, 28)):
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    n      = len(images)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.5, n_rows * 1.8))
    axes = np.array(axes).reshape(n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            ax  = axes[i, j]
            if idx < n:
                ax.imshow(images[idx].reshape(image_shape),
                          cmap='gray', vmin=0, vmax=1,
                          interpolation='nearest')
            ax.axis('off')
    fig.suptitle(f'Images générées – {model_name}', fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → {filepath}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



#  Data

print("\nChargement MNIST…")
X_train, Y_train, X_test, Y_test = load_mnist()

mask = (np.argmax(Y_train, axis=1) == 5) | (np.argmax(Y_train, axis=1) == 8)
X58  = torch.tensor(X_train[mask], dtype=torch.float32, device=DEVICE)
print(f"  Sous-ensemble (5 et 8) : {X58.shape}")

loader = DataLoader(TensorDataset(X58), batch_size=BATCH_SIZE, shuffle=True)



#  1. RBM


print("  1. RBM")


rbm = init_RBM(N_VISIBLE, HIDDEN)
rbm, _ = train_RBM(rbm, X58, epochs=N_EPOCHS_RBM,
                    learning_rate=0.01, batch_size=BATCH_SIZE, verbose=False)
gen_rbm = generer_image_RBM(rbm, N_GIBBS, N_GEN, image_shape=(28, 28))
plot_generated(gen_rbm, 'RBM',
               os.path.join(OUT, 'bonus_RBM_generated.png'),
               image_shape=(28, 28))
n_params_rbm = N_VISIBLE * HIDDEN + N_VISIBLE + HIDDEN
print(f"  Paramètres : {n_params_rbm:,}")



#  2. DBN  [784 → 256 → 64]


print("  2. DBN  [784 → 256 → 64]")


dbn = init_DBN([N_VISIBLE, HIDDEN, LATENT_DIM])
dbn, _ = train_DBN(dbn, X58, epochs=N_EPOCHS_RBM,
                    learning_rate=0.01, batch_size=BATCH_SIZE, verbose=False)
gen_dbn = generer_image_DBN(dbn, N_GIBBS, N_GEN, image_shape=(28, 28))
plot_generated(gen_dbn, 'DBN  [784→256→64]',
               os.path.join(OUT, 'bonus_DBN_generated.png'),
               image_shape=(28, 28))
n_params_dbn = (N_VISIBLE*HIDDEN + N_VISIBLE + HIDDEN +
                HIDDEN*LATENT_DIM + HIDDEN + LATENT_DIM)
print(f"  Paramètres : {n_params_dbn:,}")



#  3. VAE  (PyTorch nn.Module)


print("  3. VAE  (PyTorch)")



class VAE(nn.Module):
    def __init__(self, n_vis=784, n_hid=256, n_lat=64):
        super().__init__()
        # encoder
        self.enc_fc1 = nn.Linear(n_vis, n_hid)
        self.enc_mu  = nn.Linear(n_hid, n_lat)
        self.enc_lv  = nn.Linear(n_hid, n_lat)
        # decoder
        self.dec_fc1 = nn.Linear(n_lat, n_hid)
        self.dec_fc2 = nn.Linear(n_hid, n_vis)

    def encode(self, x):
        h      = F.relu(self.enc_fc1(x))
        return self.enc_mu(h), self.enc_lv(h)

    def reparameterise(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        return torch.sigmoid(self.dec_fc2(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z  = self.reparameterise(mu, log_var)
        xr = self.decode(z)
        return xr, mu, log_var

    @torch.no_grad()
    def generate(self, n):
        z = torch.randn(n, self.enc_mu.out_features, device=DEVICE)
        return self.decode(z)


def vae_loss(xr, x, mu, log_var):
    bce = F.binary_cross_entropy(xr, x, reduction='sum') / x.shape[0]
    kl  = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1).mean()
    return bce + kl


vae       = VAE(N_VISIBLE, HIDDEN, LATENT_DIM).to(DEVICE)
opt_vae   = torch.optim.Adam(vae.parameters(), lr=LR)
vae_losses = []

print("  Entraînement VAE…")
for epoch in range(N_EPOCHS_DL):
    ep_loss = 0.0
    for (Xb,) in loader:
        opt_vae.zero_grad()
        xr, mu, lv = vae(Xb)
        loss = vae_loss(xr, Xb, mu, lv)
        loss.backward()
        opt_vae.step()
        ep_loss += loss.item() * Xb.shape[0]
    ep_loss /= X58.shape[0]
    vae_losses.append(ep_loss)
    if (epoch + 1) % 20 == 0:
        print(f"    Epoch {epoch+1}/{N_EPOCHS_DL}  –  ELBO: {ep_loss:.4f}")

gen_vae = vae.generate(N_GEN).cpu().numpy()
plot_generated(gen_vae, 'VAE  (784→256→64→256→784)',
               os.path.join(OUT, 'bonus_VAE_generated.png'),
               image_shape=(28, 28))
n_params_vae = count_params(vae)
print(f"  Paramètres : {n_params_vae:,}")



#  4. GAN  (PyTorch nn.Module)


print("  4. GAN  (PyTorch)")
class Generator(nn.Module):
    def __init__(self, n_noise=64, n_hid=256, n_out=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_noise, n_hid), nn.ReLU(),
            nn.Linear(n_hid,  n_out),  nn.Sigmoid()
        )
    def forward(self, z): return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, n_in=784, n_hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hid), nn.LeakyReLU(0.2),
            nn.Linear(n_hid, 1),    nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)


G       = Generator(LATENT_DIM, HIDDEN, N_VISIBLE).to(DEVICE)
D       = Discriminator(N_VISIBLE, HIDDEN).to(DEVICE)
opt_G   = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D   = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
bce_gan = nn.BCELoss()

g_losses, d_losses = [], []
print("  Entraînement GAN…")

for epoch in range(N_EPOCHS_DL):
    ep_g, ep_d, nb = 0.0, 0.0, 0
    for (Xreal,) in loader:
        mb     = Xreal.shape[0]
        ones   = torch.ones(mb,  1, device=DEVICE)
        zeros  = torch.zeros(mb, 1, device=DEVICE)
        z      = torch.randn(mb, LATENT_DIM, device=DEVICE)

        # ── discriminator ─────────────────────────────────────────────────────
        opt_D.zero_grad()
        loss_D = (bce_gan(D(Xreal), ones) +
                  bce_gan(D(G(z).detach()), zeros)) * 0.5
        loss_D.backward(); opt_D.step()

        # ── generator ─────────────────────────────────────────────────────────
        opt_G.zero_grad()
        loss_G = bce_gan(D(G(z)), ones)
        loss_G.backward(); opt_G.step()

        ep_d += loss_D.item(); ep_g += loss_G.item(); nb += 1

    g_losses.append(ep_g / nb)
    d_losses.append(ep_d / nb)
    if (epoch + 1) % 20 == 0:
        print(f"    Epoch {epoch+1}/{N_EPOCHS_DL}  –  "
              f"D={ep_d/nb:.4f}  G={ep_g/nb:.4f}")

with torch.no_grad():
    z       = torch.randn(N_GEN, LATENT_DIM, device=DEVICE)
    gen_gan = G(z).cpu().numpy()

plot_generated(gen_gan, 'GAN',
               os.path.join(OUT, 'bonus_GAN_generated.png'),
               image_shape=(28, 28))
n_params_gan = count_params(G) + count_params(D)
print(f"  Paramètres : {n_params_gan:,}")



#  Comparaison finale

print("\n  Grille comparative…")

all_models   = ['RBM', 'DBN', 'VAE', 'GAN']
all_images   = [gen_rbm, gen_dbn, gen_vae, gen_gan]
all_n_params = [n_params_rbm, n_params_dbn, n_params_vae, n_params_gan]

fig, axes = plt.subplots(4, N_GEN, figsize=(N_GEN * 1.5, 4 * 2.0))
for row, (name, imgs, npar) in enumerate(
        zip(all_models, all_images, all_n_params)):
    for col in range(N_GEN):
        ax = axes[row, col]
        if col < len(imgs):
            ax.imshow(imgs[col].reshape(28, 28),
                      cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(f'{name}\n({npar:,} params)',
                          fontsize=8, rotation=0, labelpad=75, va='center')

plt.suptitle('Comparaison des modèles génératifs – MNIST (2 et 3)',
             fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'bonus_comparison_grid.png'),
            dpi=130, bbox_inches='tight')
plt.close()
print(f"  → {os.path.join(OUT, 'bonus_comparison_grid.png')}")

# training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(vae_losses, color='steelblue', lw=1.6)
axes[0].set_title('VAE – ELBO loss'); axes[0].set_xlabel('Epoch')
axes[0].grid(True, ls='--', alpha=0.5)

axes[1].plot(g_losses, color='steelblue',  lw=1.6, label='Generator')
axes[1].plot(d_losses, color='darkorange', lw=1.6, ls='--', label='Discriminator')
axes[1].set_title('GAN – Adversarial losses'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(True, ls='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'bonus_training_curves.png'), dpi=130)
plt.close()
print(f"  → {os.path.join(OUT, 'bonus_training_curves.png')}")

# param count bar
fig, ax = plt.subplots(figsize=(7, 4))
colors = ['steelblue', 'darkorange', 'seagreen', 'tomato']
bars   = ax.bar(all_models, [p/1e3 for p in all_n_params],
                color=colors, edgecolor='black', alpha=0.85)
ax.set_ylabel('Paramètres (×1 000)')
ax.set_title('Nombre de paramètres par modèle')
ax.grid(True, axis='y', ls='--', alpha=0.5)
for bar, npar in zip(bars, all_n_params):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1, f'{npar:,}',
            ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'bonus_param_count.png'), dpi=130)
plt.close()
print(f"  → {os.path.join(OUT, 'bonus_param_count.png')}")

print(f"\nRésultats sauvegardés dans '{OUT}/'")
print("Terminé.\n")