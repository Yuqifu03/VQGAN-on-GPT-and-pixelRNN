# VQ-GAN + Transformer (GPT-style) on Fashion-MNIST
# --------------------------------------------------
# End-to-end training script:
# 1) Train a (lightweight) VQ-GAN on Fashion-MNIST (28x28 grayscale)
# 2) Encode the dataset into codebook indices (latent tokens)
# 3) Train a causal Transformer over latent code sequences
# 4) Sample new sequences and decode via VQ-GAN decoder to synthesize images
#
# Usage (typical):
#   python vqgan_transformer_fashionmnist.py --stage vqgan --epochs 10
#   python vqgan_transformer_fashionmnist.py --stage encode
#   python vqgan_transformer_fashionmnist.py --stage gpt --epochs 10
#   python vqgan_transformer_fashionmnist.py --stage sample --num_samples 16
#
# Tips:
#  - Start with small epochs (e.g., 5-10) to validate the pipeline, then scale up.
#  - GPU is highly recommended. Set --device cuda if available.
#  - You can disable the GAN discriminator (pure VQ-VAE) with --no_gan for stability.

import os
import math
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils

# ----------------------
# Utility helpers
# ----------------------

def exists(x):
    return x is not None


def default(val, d):
    return d if val is None else val


def save_image_grid(tensor, path, nrow=4, normalize=True, value_range=(-1, 1)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_image(tensor, path, nrow=nrow, normalize=normalize, value_range=value_range)

# ----------------------
# Vector Quantizer (non-EMA)
# ----------------------

class WassersteinQuantizer(nn.Module):
    """Wasserstein VQ to reduce codebook collapse."""
    def __init__(self, n_codes=512, code_dim=64, beta=0.25, w_decay=0.1):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.beta = beta
        self.w_decay = w_decay
        self.codebook = nn.Embedding(n_codes, code_dim)
        nn.init.xavier_uniform_(self.codebook.weight)

    def forward(self, z):
        b, c, h, w = z.shape
        z_perm = z.permute(0,2,3,1).contiguous()
        z_flat = z_perm.view(-1, c)
        e = self.codebook.weight
        # earth-mover like soft assignment
        d = torch.cdist(z_flat.unsqueeze(0), e.unsqueeze(0), p=2).squeeze(0)
        soft = F.softmax(-d, dim=1)
        z_q = soft @ e
        z_q = z_q.view(b,h,w,c).permute(0,3,1,2).contiguous()
        z_q_st = z + (z_q - z).detach()
        # compute indices for saving codes
        indices = soft.argmax(dim=1).view(b,h,w)
        # losses
        commit = F.mse_loss(z.detach(), z_q)
        cb = F.mse_loss(z, z_q.detach())
        reg = self.w_decay * soft.var(dim=1).mean()
        vq_loss = cb + self.beta * commit + reg
        return z_q_st, vq_loss, indices

class VectorQuantizer(nn.Module):
    """Basic VQ with straight-through estimator.
    Args:
        n_codes: codebook size (vocabulary)
        code_dim: embedding dimension per code
        beta: commitment loss weight
    """
    def __init__(self, n_codes=512, code_dim=64, beta=0.25):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.beta = beta
        self.codebook = nn.Embedding(n_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / n_codes, 1.0 / n_codes)

    def forward(self, z):
        # z: (b, c, h, w)
        b, c, h, w = z.shape
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)
        z_flat = z_perm.view(-1, c)  # (b*h*w, c)

        # Compute distances to codebook
        codebook_weight = self.codebook.weight  # (n_codes, c)
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z.e
        z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)  # (bhw, 1)
        e_sq = (codebook_weight ** 2).sum(dim=1)       # (n_codes)
        ze = z_flat @ codebook_weight.t()              # (bhw, n_codes)
        dists = z_sq + e_sq[None, :] - 2 * ze

        # nearest code
        indices = dists.argmin(dim=1)                  # (bhw)
        z_q = self.codebook(indices).view(b, h, w, c)  # (b, h, w, c)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()     # (b, c, h, w)

        # straight-through
        z_q_st = z + (z_q - z).detach()

        # losses
        # commitment: ||z.detach() - z_q||^2
        commitment = F.mse_loss(z.detach(), z_q)
        # codebook: ||z - z_q.detach()||^2
        codebook = F.mse_loss(z, z_q.detach())
        vq_loss = codebook + self.beta * commitment

        return z_q_st, vq_loss, indices.view(b, h, w)

# ----------------------
# VQ-GAN (lite)
# ----------------------

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
    def __init__(self, in_ch=1, base=64, code_dim=64):
        super().__init__()
        # Downsample x2 twice: 28x28 -> 7x7 (factor 4)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            ResBlock(base),
            nn.Conv2d(base, base, 4, stride=2, padding=1),  # 14x14
            ResBlock(base),
            nn.Conv2d(base, base*2, 4, stride=2, padding=1), # 7x7
            ResBlock(base*2),
            nn.GroupNorm(8, base*2),
            nn.SiLU(),
            nn.Conv2d(base*2, code_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_ch=1, base=64, code_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(code_dim, base*2, 1),
            nn.GroupNorm(8, base*2),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            ResBlock(base*2),
            nn.ConvTranspose2d(base*2, base, 4, stride=2, padding=1),  # 14x14
            ResBlock(base),
            nn.ConvTranspose2d(base, base, 4, stride=2, padding=1),    # 28x28
            ResBlock(base),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, out_ch, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z_q):
        z = self.proj(z_q)
        return self.net(z)


class Discriminator(nn.Module):
    """Very small PatchDiscriminator. Can be disabled for VQ-VAE-only training."""
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*2, base*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*4, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class VQGAN(nn.Module):
    def __init__(self, in_ch=1, code_dim=64, n_codes=512, beta=0.25, gan_weight=0.8, use_gan=True):
        super().__init__()
        self.encoder = Encoder(in_ch=in_ch, code_dim=code_dim)
        self.quantizer = WassersteinQuantizer(n_codes=n_codes, code_dim=code_dim, beta=beta)
        self.decoder = Decoder(out_ch=in_ch, code_dim=code_dim)
        self.use_gan = use_gan
        self.gan_weight = gan_weight
        if use_gan:
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

# ----------------------
# GPT-style Transformer for latent tokens
# ----------------------

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
        qkv = self.qkv(x)  # (b, n, 3d)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)  # (b, h, n, d_h)
        k = k.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (b, h, n, n)
        # causal mask
        mask = torch.triu(torch.ones(n, n, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (b, h, n, d_h)
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
        # idx: (b, n) int64
        b, n = idx.shape
        assert n <= self.seq_len, "Sequence longer than model's maximum context length"
        x = self.token_emb(idx) + self.pos_emb[:, :n, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (b, n, vocab)
        return logits

    @torch.no_grad()
    def generate(self, idx, steps, temperature=1.0, top_k=None):
        # idx: (b, cur_len)
        for _ in range(steps):
            idx_cond = idx[:, -self.seq_len:]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if exists(top_k):
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

# ----------------------
# Data & Tokenization utils
# ----------------------

def get_dataloaders(batch_size=128, root='./data'):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1, 1]
    ])
    train_set = datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
    test_set = datasets.FashionMNIST(root, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def encode_dataset_to_codes(model: VQGAN, loader, device, save_path):
    model.eval()
    all_indices = []
    for x, _ in loader:
        x = x.to(device)
        z = model.encoder(x)
        _, _, indices = model.quantizer(z)
        # indices: (b, h, w)
        all_indices.append(indices.cpu())
    codes = torch.cat(all_indices, dim=0)  # (N, h, w)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(codes, save_path)
    # also save shape meta
    meta = {
        'n_codes': model.quantizer.n_codes,
        'code_dim': model.quantizer.code_dim,
        'latent_hw': codes.shape[1:3]
    }
    torch.save(meta, save_path.with_suffix('.meta.pt'))
    print(f"Saved codes to {save_path} with shape {tuple(codes.shape)} and meta {meta}")


class CodeSequenceDataset(Dataset):
    def __init__(self, codes_tensor: torch.Tensor):
        # codes_tensor: (N, h, w), ints in [0, vocab)
        self.codes = codes_tensor
        self.h, self.w = codes_tensor.shape[1:]
        self.seq_len = self.h * self.w

    def __len__(self):
        return self.codes.shape[0]

    def __getitem__(self, idx):
        grid = self.codes[idx]  # (h, w)
        seq = grid.flatten()    # (h*w,)
        # inputs are seq[:-1], targets are seq[1:]
        x = seq[:-1]
        y = seq[1:]
        return x.long(), y.long()


def get_code_dataloaders(code_path, batch_size=256):
    codes = torch.load(code_path)  # (N, h, w)
    ds = CodeSequenceDataset(codes)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return loader, ds.seq_len

# ----------------------
# Training loops
# ----------------------

def train_vqgan(args):
    device = torch.device(args.device)
    train_loader, test_loader = get_dataloaders(args.batch_size, args.data_root)

    model = VQGAN(
        in_ch=1,
        code_dim=args.code_dim,
        n_codes=args.n_codes,
        beta=args.beta,
        gan_weight=args.gan_weight,
        use_gan=not args.no_gan,
    ).to(device)

    if args.ckpt_vqgan and Path(args.ckpt_vqgan).exists():
        model.load_state_dict(torch.load(args.ckpt_vqgan, map_location=device))
        print(f"Loaded VQ-GAN from {args.ckpt_vqgan}")

    opt_g = torch.optim.Adam(list(model.encoder.parameters()) +
                             list(model.quantizer.parameters()) +
                             list(model.decoder.parameters()), lr=args.lr_vq)

    if model.use_gan:
        opt_d = torch.optim.Adam(model.disc.parameters(), lr=args.lr_disc)
        bce = nn.BCEWithLogitsLoss()

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for x, _ in train_loader:
            x = x.to(device)
            # --- generator (autoencoder + vq) step ---
            x_rec, vq_loss, _ = model(x)
            rec_loss = F.l1_loss(x_rec, x)
            g_loss = rec_loss + vq_loss

            if model.use_gan:
                logits_fake = model.disc(x_rec)
                valid = torch.ones_like(logits_fake)
                adv_loss = bce(logits_fake, valid)
                g_loss = g_loss + args.gan_weight * adv_loss

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            # --- discriminator step ---
            if model.use_gan:
                with torch.no_grad():
                    x_rec_detach = model(x)[0]
                logits_real = model.disc(x)
                logits_fake = model.disc(x_rec_detach)
                valid = torch.ones_like(logits_real)
                fake = torch.zeros_like(logits_fake)
                d_loss = bce(logits_real, valid) + bce(logits_fake, fake)
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
            else:
                d_loss = torch.tensor(0.0, device=device)

            if global_step % args.log_every == 0:
                print(f"[VQGAN] epoch {epoch} step {global_step} rec {rec_loss.item():.4f} vq {vq_loss.item():.4f} d {d_loss.item():.4f}")
            if global_step % args.sample_every == 0:
                with torch.no_grad():
                    save_image_grid(torch.cat([x[:8], x_rec[:8]], dim=0), f"{args.out_dir}/vqgan_rec_e{epoch}_s{global_step}.png", nrow=8)
            global_step += 1

        # save checkpoint each epoch
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), f"{args.out_dir}/vqgan_epoch{epoch}.pt")

    # save final
    torch.save(model.state_dict(), f"{args.out_dir}/vqgan_final.pt")


def encode_stage(args):
    device = torch.device(args.device)
    _, test_loader = get_dataloaders(args.batch_size, args.data_root)  # use train set for codes
    train_loader, _ = get_dataloaders(args.batch_size, args.data_root)

    model = VQGAN(
        in_ch=1,
        code_dim=args.code_dim,
        n_codes=args.n_codes,
        beta=args.beta,
        gan_weight=args.gan_weight,
        use_gan=False,
    ).to(device)
    assert args.ckpt_vqgan and Path(args.ckpt_vqgan).exists(), "Provide trained VQ-GAN checkpoint via --ckpt_vqgan"
    state = torch.load(args.ckpt_vqgan, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # encode TRAIN split to tokens (more data for GPT)
    encode_dataset_to_codes(model, train_loader, device, f"{args.out_dir}/train_codes.pt")
    # also encode TEST split (optional)
    encode_dataset_to_codes(model, test_loader, device, f"{args.out_dir}/test_codes.pt")


def train_gpt(args):
    device = torch.device(args.device)
    train_loader, seq_len = get_code_dataloaders(args.train_codes, batch_size=args.batch_size)
    meta = torch.load(Path(args.train_codes).with_suffix('.meta.pt'))
    vocab = meta['n_codes']

    # sequence for teacher-forcing is (seq_len-1)
    model = GPTTokens(
        vocab_size=vocab,
        seq_len=seq_len - 1,
        dim=args.gpt_dim,
        n_layers=args.gpt_layers,
        n_heads=args.gpt_heads,
        dropout=args.gpt_dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_gpt, betas=(0.9, 0.95), weight_decay=0.01)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if global_step % args.log_every == 0:
                print(f"[GPT] epoch {epoch} step {global_step} loss {loss.item():.4f}")
            global_step += 1

        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), f"{args.out_dir}/gpt_epoch{epoch}.pt")

    torch.save(model.state_dict(), f"{args.out_dir}/gpt_final.pt")


@torch.no_grad()
def sample_stage(args):
    device = torch.device(args.device)

    # load VQ-GAN (decoder + codebook)
    vq = VQGAN(
        in_ch=1,
        code_dim=args.code_dim,
        n_codes=args.n_codes,
        beta=args.beta,
        gan_weight=args.gan_weight,
        use_gan=False,
    ).to(device)
    assert args.ckpt_vqgan and Path(args.ckpt_vqgan).exists()
    state = torch.load(args.ckpt_vqgan, map_location=device)
    vq.load_state_dict(state, strict=False)
    vq.eval()

    # load meta for latent grid size
    meta = torch.load(Path(args.train_codes).with_suffix('.meta.pt'))
    h, w = meta['latent_hw']
    seq_len = h * w
    vocab = meta['n_codes']

    # load GPT
    gpt = GPTTokens(vocab_size=vocab, seq_len=seq_len - 1, dim=args.gpt_dim,
                    n_layers=args.gpt_layers, n_heads=args.gpt_heads, dropout=args.gpt_dropout).to(device)
    assert args.ckpt_gpt and Path(args.ckpt_gpt).exists()
    gpt.load_state_dict(torch.load(args.ckpt_gpt, map_location=device))
    gpt.eval()

    # Autoregressive sampling for token sequences
    b = args.num_samples
    # start token: use a dummy first token; here we just pick zeros
    start = torch.zeros(b, 1, dtype=torch.long, device=device)
    seq = gpt.generate(start, steps=seq_len-1, temperature=args.temperature, top_k=args.top_k)
    seq = seq[:, 1:]  # drop the start token, now length seq_len-1

    # prepend one more token (e.g., zero) to make exact seq_len
    pad = torch.zeros(b, 1, dtype=torch.long, device=device)
    full_seq = torch.cat([pad, seq], dim=1)  # (b, seq_len)

    # reshape to (b, h, w)
    grids = full_seq.view(b, h, w)

    # turn indices -> embeddings -> decode
    emb = vq.quantizer.codebook(grids.view(b, -1))  # (b, h*w, code_dim)
    z_q = emb.view(b, h, w, vq.quantizer.code_dim).permute(0, 3, 1, 2).contiguous()
    x_gen = vq.decode(z_q)

    save_image_grid(x_gen, f"{args.out_dir}/samples.png", nrow=int(math.sqrt(b)))
    print(f"Saved samples to {args.out_dir}/samples.png")

# ----------------------
# Main
# ----------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=str, choices=['vqgan', 'encode', 'gpt', 'sample'], required=True)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--out_dir', type=str, default='./runs')

    # VQ-GAN
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--code_dim', type=int, default=64)
    p.add_argument('--n_codes', type=int, default=512)
    p.add_argument('--beta', type=float, default=0.25)
    p.add_argument('--gan_weight', type=float, default=0.8)
    p.add_argument('--no_gan', action='store_true')
    p.add_argument('--lr_vq', type=float, default=2e-4)
    p.add_argument('--lr_disc', type=float, default=2e-4)
    p.add_argument('--ckpt_vqgan', type=str, default='')

    # GPT
    p.add_argument('--train_codes', type=str, default='./runs/train_codes.pt')
    p.add_argument('--gpt_dim', type=int, default=256)
    p.add_argument('--gpt_layers', type=int, default=6)
    p.add_argument('--gpt_heads', type=int, default=8)
    p.add_argument('--gpt_dropout', type=float, default=0.1)
    p.add_argument('--lr_gpt', type=float, default=3e-4)
    p.add_argument('--ckpt_gpt', type=str, default='')

    # misc
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--sample_every', type=int, default=500)

    # sampling
    p.add_argument('--num_samples', type=int, default=16)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    if args.stage == 'vqgan':
        train_vqgan(args)
    elif args.stage == 'encode':
        encode_stage(args)
    elif args.stage == 'gpt':
        train_gpt(args)
    elif args.stage == 'sample':
        sample_stage(args)


if __name__ == '__main__':
    main()
  
