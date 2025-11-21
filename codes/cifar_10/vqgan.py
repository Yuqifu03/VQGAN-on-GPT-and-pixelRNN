"""
VQ-(VAE/GAN) + GPT on CIFAR-10
------------------------------

Stages:
  1) Train a vector-quantized autoencoder (VQ-VAE or VQ-GAN) on CIFAR-10.
  2) Encode CIFAR-10 into latent code indices.
  3) Train a GPT-style Transformer on sequences of code indices.
  4) Sample new latent sequences with GPT and decode to images.

Usage (examples):

  # Train VQ-VAE
  python vq_model_cifar10.py --stage vq --model_type vqvae --epochs 20

  # Train VQ-GAN
  python vq_model_cifar10.py --stage vq --model_type vqgan --epochs 20

  # Encode CIFAR-10 codes with VQ-VAE model
  python vq_model_cifar10.py --stage encode \
      --model_type vqvae \
      --ckpt_vq runs/vqvae_final.pt

  # Train GPT on VQ-VAE codes
  python vq_model_cifar10.py --stage gpt \
      --train_codes runs/train_codes_vqvae.pt --epochs 20

  # Sample from VQ-VAE + GPT
  python vq_model_cifar10.py --stage sample \
      --model_type vqvae \
      --ckpt_vq runs/vqvae_final.pt \
      --ckpt_gpt runs/gpt_final_vqvae.pt \
      --train_codes runs/train_codes_vqvae.pt \
      --num_samples 16
"""

import os
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def exists(x):
    return x is not None


def save_image_grid(tensor, path, nrow=4, normalize=True, value_range=(-1, 1)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_image(tensor, path, nrow=nrow, normalize=normalize, value_range=value_range)
    print(f"[save_image_grid] Saved image grid to {path}")


def seed_everything(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Vector Quantizer (non-EMA)
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """Basic VQ with straight-through estimator.

    Args:
        n_codes: codebook size (vocabulary)
        code_dim: embedding dimension per code
        beta: commitment loss weight
    """
    def __init__(self, n_codes=512, code_dim=128, beta=0.25):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.beta = beta
        self.codebook = nn.Embedding(n_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / n_codes, 1.0 / n_codes)

    def forward(self, z):
        """
        z: (B, C, H, W)
        Returns:
            z_q_st: quantized latent with straight-through gradient (B, C, H, W)
            vq_loss: scalar
            indices: discrete code indices (B, H, W)
        """
        b, c, h, w = z.shape
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        z_flat = z_perm.view(-1, c)                  # (B*H*W, C)

        # Compute squared distances to codebook: ||z - e||^2
        codebook_weight = self.codebook.weight       # (K, C)
        z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)        # (BHW, 1)
        e_sq = (codebook_weight ** 2).sum(dim=1)              # (K,)
        ze = z_flat @ codebook_weight.t()                     # (BHW, K)
        dists = z_sq + e_sq[None, :] - 2 * ze                 # (BHW, K)

        indices = dists.argmin(dim=1)                         # (BHW,)
        z_q = self.codebook(indices).view(b, h, w, c)         # (B, H, W, C)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()            # (B, C, H, W)

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        # VQ losses
        commitment = F.mse_loss(z.detach(), z_q)      # encoder commits to code
        codebook = F.mse_loss(z, z_q.detach())        # update codebook
        vq_loss = codebook + self.beta * commitment

        return z_q_st, vq_loss, indices.view(b, h, w)


# ---------------------------------------------------------------------------
# VQ Autoencoder (VQ-VAE / VQ-GAN)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    """
    CIFAR-10 encoder:
      Input:  (B, 3, 32, 32)
      Output: (B, code_dim, 8, 8)
    """
    def __init__(self, in_ch=3, base=64, code_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),           # 32x32
            ResBlock(base),
            nn.Conv2d(base, base, 4, stride=2, padding=1),  # 32→16
            ResBlock(base),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),  # 16→8
            ResBlock(base * 2),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
            nn.Conv2d(base * 2, code_dim, 1)                # (B, code_dim, 8, 8)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """
    CIFAR-10 decoder:
      Input:  (B, code_dim, 8, 8)
      Output: (B, 3, 32, 32)
    """
    def __init__(self, out_ch=3, base=64, code_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(code_dim, base * 2, 1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            ResBlock(base * 2),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),  # 8→16
            ResBlock(base),
            nn.ConvTranspose2d(base, base, 4, stride=2, padding=1),      # 16→32
            ResBlock(base),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, out_ch, 3, padding=1),
            nn.Tanh(),  # outputs in [-1,1]
        )

    def forward(self, z_q):
        z = self.proj(z_q)
        return self.net(z)


class Discriminator(nn.Module):
    """Patch discriminator used only for VQ-GAN (not VQ-VAE)."""
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, stride=2, padding=1),   # 32→16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),  # 16→8
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),  # 8→4
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, 1, 3, padding=1)  # (B,1,H',W')
        )

    def forward(self, x):
        return self.net(x)


class VQAutoEncoder(nn.Module):
    """
    Unifies VQ-VAE and VQ-GAN:

    model_type:
      - 'vqvae' → no discriminator, purely reconstruction + VQ loss
      - 'vqgan' → adds adversarial loss from discriminator
    """
    def __init__(
        self,
        in_ch=3,
        code_dim=128,
        n_codes=512,
        beta=0.25,
        model_type='vqgan',
        gan_weight=0.8
    ):
        super().__init__()
        assert model_type in ['vqgan', 'vqvae']
        self.model_type = model_type

        self.encoder = Encoder(in_ch=in_ch, code_dim=code_dim)
        self.quantizer = VectorQuantizer(n_codes=n_codes, code_dim=code_dim, beta=beta)
        self.decoder = Decoder(out_ch=in_ch, code_dim=code_dim)

        self.use_gan = (model_type == 'vqgan')
        self.gan_weight = gan_weight

        if self.use_gan:
            self.disc = Discriminator(in_ch=in_ch)

    def encode(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z)
        return z, z_q, vq_loss, indices

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z)
        x_rec = self.decoder(z_q)
        return x_rec, vq_loss, indices


# ---------------------------------------------------------------------------
# GPT-style Transformer for latent tokens
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x)  # (B, N, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, N, Dh)
        k = k.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, N, N)

        # causal mask: cannot attend to future positions
        mask = torch.triu(torch.ones(n, n, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v                         # (B, H, N, Dh)
        y = y.transpose(1, 2).contiguous().view(b, n, d)
        y = self.resid_drop(self.proj(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTTokens(nn.Module):
    def __init__(self, vocab_size, seq_len, dim=256, n_layers=6, n_heads=8, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.seq_len = seq_len

    def forward(self, idx):
        """
        idx: (B, N) int64
        Returns:
            logits: (B, N, vocab_size)
        """
        b, n = idx.shape
        assert n <= self.seq_len, "Sequence longer than model's maximum context length"
        x = self.token_emb(idx) + self.pos_emb[:, :n, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, steps, temperature=1.0, top_k=None):
        """
        Autoregressive generation starting from idx.
        idx: (B, cur_len)
        steps: number of new tokens to generate
        """
        for _ in range(steps):
            idx_cond = idx[:, -self.seq_len:]           # ensure within context
            logits = self.forward(idx_cond)             # (B, L, V)
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # last token

            if exists(top_k):
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# ---------------------------------------------------------------------------
# CIFAR-10 data & tokenization
# ---------------------------------------------------------------------------

def get_dataloaders(batch_size=128, root='./data'):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # normalize to [-1,1]
    ])
    train_set = datasets.CIFAR10(root, train=True, download=True, transform=tfm)
    test_set = datasets.CIFAR10(root, train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def encode_dataset_to_codes(model: VQAutoEncoder, loader, device, save_path: str):
    model.eval()
    all_indices = []
    for x, _ in loader:
        x = x.to(device)
        z = model.encoder(x)
        _, _, indices = model.quantizer(z)  # (B, H, W)
        all_indices.append(indices.cpu())
    codes = torch.cat(all_indices, dim=0)  # (N, H, W)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(codes, save_path)

    meta = {
        'n_codes': model.quantizer.n_codes,
        'code_dim': model.quantizer.code_dim,
        'latent_hw': codes.shape[1:3],   # (H, W)
    }
    meta_path = save_path.with_suffix('.meta.pt')
    torch.save(meta, meta_path)
    print(f"[encode_dataset_to_codes] Saved codes to {save_path} with shape {tuple(codes.shape)}")
    print(f"[encode_dataset_to_codes] Saved meta to {meta_path}: {meta}")


class CodeSequenceDataset(Dataset):
    """
    Turn code grids (N, H, W) into sequences of length L = H*W, with
    teacher-forcing pairs (x = seq[:-1], y = seq[1:]).
    """
    def __init__(self, codes_tensor: torch.Tensor):
        super().__init__()
        self.codes = codes_tensor
        self.h, self.w = codes_tensor.shape[1:]
        self.seq_len = self.h * self.w

    def __len__(self):
        return self.codes.shape[0]

    def __getitem__(self, idx):
        grid = self.codes[idx]          # (H, W)
        seq = grid.flatten()            # (L,)
        x = seq[:-1]                    # (L-1,)
        y = seq[1:]                     # (L-1,)
        return x.long(), y.long()


def get_code_dataloader(code_path: str, batch_size=256):
    codes = torch.load(code_path)  # (N, H, W)
    ds = CodeSequenceDataset(codes)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)
    return loader, ds.seq_len  # seq_len = H*W


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_vq(args):
    """
    Train VQ-VAE or VQ-GAN autoencoder on CIFAR-10.
    """
    device = torch.device(args.device)
    seed_everything(args.seed)

    train_loader, _ = get_dataloaders(args.batch_size, args.data_root)

    model = VQAutoEncoder(
        in_ch=3,
        code_dim=args.code_dim,
        n_codes=args.n_codes,
        beta=args.beta,
        model_type=args.model_type,
        gan_weight=args.gan_weight
    ).to(device)

    if args.ckpt_vq and Path(args.ckpt_vq).exists():
        model.load_state_dict(torch.load(args.ckpt_vq, map_location=device))
        print(f"[train_vq] Loaded VQ model from {args.ckpt_vq}")

    opt_g = torch.optim.Adam(
        list(model.encoder.parameters()) +
        list(model.quantizer.parameters()) +
        list(model.decoder.parameters()),
        lr=args.lr_vq
    )

    if model.use_gan:
        opt_d = torch.optim.Adam(model.disc.parameters(), lr=args.lr_disc)
        bce = nn.BCEWithLogitsLoss()

    global_step = 0
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for x, _ in train_loader:
            x = x.to(device)

            # --- Generator: reconstruction + VQ (+ adversarial if VQ-GAN) ---
            x_rec, vq_loss, _ = model(x)
            rec_loss = F.l1_loss(x_rec, x)
            g_loss = rec_loss + vq_loss

            if model.use_gan:
                logits_fake = model.disc(x_rec)
                valid = torch.ones_like(logits_fake)
                adv_loss = bce(logits_fake, valid)
                g_loss = g_loss + args.gan_weight * adv_loss
            else:
                adv_loss = torch.tensor(0.0, device=device)

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            # --- Discriminator (VQ-GAN only) ---
            if model.use_gan:
                with torch.no_grad():
                    x_fake = model(x)[0]
                logits_real = model.disc(x)
                logits_fake = model.disc(x_fake)
                valid = torch.ones_like(logits_real)
                fake = torch.zeros_like(logits_fake)
                d_loss = bce(logits_real, valid) + bce(logits_fake, fake)

                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
            else:
                d_loss = torch.tensor(0.0, device=device)

            if global_step % args.log_every == 0:
                print(
                    f"[VQ-{args.model_type}] epoch {epoch} "
                    f"step {global_step} rec {rec_loss.item():.4f} "
                    f"vq {vq_loss.item():.4f} adv {adv_loss.item():.4f} "
                    f"d {d_loss.item():.4f}"
                )

            if global_step % args.sample_every == 0 and global_step%40==0 or global_step==args.epochs-1:
                with torch.no_grad():
                    save_image_grid(
                        torch.cat([x[:8], x_rec[:8]], dim=0),
                        f"{args.out_dir}/rec_{args.model_type}_e{epoch}_s{global_step}.png",
                        nrow=8
                    )

            global_step += 1

        # save checkpoint each epoch
        # ckpt_epoch = Path(args.out_dir) / f"vq_{args.model_type}_epoch{epoch}.pt"
        # torch.save(model.state_dict(), ckpt_epoch)
        # print(f"[train_vq] Saved checkpoint: {ckpt_epoch}")

    # save final checkpoint
    ckpt_final = Path(args.out_dir) / f"vq_{args.model_type}_final.pt"
    torch.save(model.state_dict(), ckpt_final)
    print(f"[train_vq] Saved final checkpoint: {ckpt_final}")


def encode_stage(args):
    """
    Encode CIFAR-10 images into latent code indices using a trained VQ autoencoder.
    """
    device = torch.device(args.device)
    seed_everything(args.seed)

    train_loader, test_loader = get_dataloaders(args.batch_size, args.data_root)

    model = VQAutoEncoder(
        in_ch=3,
        code_dim=args.code_dim,
        n_codes=args.n_codes,
        beta=args.beta,
        model_type=args.model_type,
        gan_weight=args.gan_weight
    ).to(device)

    assert args.ckpt_vq and Path(args.ckpt_vq).exists(), \
        "Provide trained VQ checkpoint via --ckpt_vq"
    state = torch.load(args.ckpt_vq, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[encode_stage] Loaded VQ model from {args.ckpt_vq}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train_codes_path = Path(args.out_dir) / f"train_codes_{args.model_type}.pt"
    test_codes_path = Path(args.out_dir) / f"test_codes_{args.model_type}.pt"

    encode_dataset_to_codes(model, train_loader, device, str(train_codes_path))
    encode_dataset_to_codes(model, test_loader, device, str(test_codes_path))


def train_gpt(args):
    """
    Train GPT on sequences of code indices.
    """
    device = torch.device(args.device)
    seed_everything(args.seed)

    train_loader, seq_len = get_code_dataloader(args.train_codes, batch_size=args.batch_size)
    meta = torch.load(Path(args.train_codes).with_suffix('.meta.pt'))
    vocab = meta['n_codes']

    model = GPTTokens(
        vocab_size=vocab,
        seq_len=seq_len - 1,   # teacher-forcing on seq[:-1] → predict seq[1:]
        dim=args.gpt_dim,
        n_layers=args.gpt_layers,
        n_heads=args.gpt_heads,
        dropout=args.gpt_dropout
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr_gpt,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)  # (B, L-1)
            y = y.to(device)  # (B, L-1)

            logits = model(x)  # (B, L-1, vocab)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if global_step % args.log_every == 0:
                print(f"[GPT] epoch {epoch} step {global_step} loss {loss.item():.4f}")
            global_step += 1

        # ckpt_epoch = Path(args.out_dir) / f"gpt_epoch{epoch}.pt"
        # torch.save(model.state_dict(), ckpt_epoch)
        # print(f"[train_gpt] Saved GPT checkpoint: {ckpt_epoch}")

    ckpt_final = Path(args.out_dir) / "gpt_final.pt"
    torch.save(model.state_dict(), ckpt_final)
    print(f"[train_gpt] Saved final GPT checkpoint: {ckpt_final}")


@torch.no_grad()
def sample_stage(args):
    """
    Sample new images using trained VQ autoencoder + GPT.
    """
    device = torch.device(args.device)
    seed_everything(args.seed)

    # Load VQ autoencoder (decoder + codebook)
    vq = VQAutoEncoder(
        in_ch=3,
        code_dim=args.code_dim,
        n_codes=args.n_codes,
        beta=args.beta,
        model_type=args.model_type,
        gan_weight=args.gan_weight
    ).to(device)

    assert args.ckpt_vq and Path(args.ckpt_vq).exists(), \
        "Provide trained VQ checkpoint via --ckpt_vq"
    state = torch.load(args.ckpt_vq, map_location=device)
    vq.load_state_dict(state, strict=False)
    vq.eval()
    print(f"[sample_stage] Loaded VQ model from {args.ckpt_vq}")

    # Load meta for latent grid size
    meta = torch.load(Path(args.train_codes).with_suffix('.meta.pt'))
    h, w = meta['latent_hw']
    seq_len = h * w
    vocab = meta['n_codes']

    # Load GPT
    gpt = GPTTokens(
        vocab_size=vocab,
        seq_len=seq_len - 1,
        dim=args.gpt_dim,
        n_layers=args.gpt_layers,
        n_heads=args.gpt_heads,
        dropout=args.gpt_dropout
    ).to(device)

    assert args.ckpt_gpt and Path(args.ckpt_gpt).exists(), \
        "Provide trained GPT checkpoint via --ckpt_gpt"
    gpt.load_state_dict(torch.load(args.ckpt_gpt, map_location=device))
    gpt.eval()
    print(f"[sample_stage] Loaded GPT model from {args.ckpt_gpt}")

    # Autoregressive sampling for token sequences
    b = args.num_samples
    start = torch.zeros(b, 1, dtype=torch.long, device=device)  # dummy start token
    seq = gpt.generate(
        start,
        steps=seq_len - 1,
        temperature=args.temperature,
        top_k=args.top_k
    )
    seq = seq[:, 1:]  # drop initial dummy token, length = seq_len-1

    # Prepend one extra (e.g., zero) to make exact seq_len
    pad = torch.zeros(b, 1, dtype=torch.long, device=device)
    full_seq = torch.cat([pad, seq], dim=1)  # (B, seq_len)

    # Reshape into (B, H, W)
    grids = full_seq.view(b, h, w)

    # indices -> embeddings -> decode
    emb = vq.quantizer.codebook(grids.view(b, -1))  # (B, H*W, code_dim)
    z_q = emb.view(b, h, w, vq.quantizer.code_dim).permute(0, 3, 1, 2).contiguous()
    x_gen = vq.decode(z_q)  # (B, 3, 32, 32)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    samples_path = Path(args.out_dir) / f"samples_{args.model_type}.png"
    save_image_grid(x_gen, samples_path, nrow=int(math.sqrt(b)))
    print(f"[sample_stage] Saved samples to {samples_path}")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--stage',
        type=str,
        choices=['vq', 'encode', 'gpt', 'sample'],
        required=True,
        help='Training stage: vq=autoencoder, encode, gpt, sample'
    )
    p.add_argument('--model_type', type=str, default='vqgan',
                  choices=['vqgan', 'vqvae'],
                  help='Autoencoder type: VQ-GAN or VQ-VAE')
    p.add_argument('--device', type=str,
                  default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--out_dir', type=str, default='./runs')
    p.add_argument('--seed', type=int, default=42)

    # VQ autoencoder
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--code_dim', type=int, default=128)
    p.add_argument('--n_codes', type=int, default=512)
    p.add_argument('--beta', type=float, default=0.25)
    p.add_argument('--gan_weight', type=float, default=0.8)
    p.add_argument('--lr_vq', type=float, default=2e-4)
    p.add_argument('--lr_disc', type=float, default=2e-4)
    p.add_argument('--ckpt_vq', type=str, default='')

    # GPT
    p.add_argument('--train_codes', type=str, default='./runs/train_codes_vqgan.pt')
    p.add_argument('--gpt_dim', type=int, default=256)
    p.add_argument('--gpt_layers', type=int, default=6)
    p.add_argument('--gpt_heads', type=int, default=8)
    p.add_argument('--gpt_dropout', type=float, default=0.1)
    p.add_argument('--lr_gpt', type=float, default=3e-4)
    p.add_argument('--ckpt_gpt', type=str, default='')

    # Logging & sampling
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--sample_every', type=int, default=500)
    p.add_argument('--num_samples', type=int, default=16)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    if args.stage == 'vq':
        train_vq(args)
    elif args.stage == 'encode':
        encode_stage(args)
    elif args.stage == 'gpt':
        train_gpt(args)
    elif args.stage == 'sample':
        sample_stage(args)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == '__main__':
    main()
  
