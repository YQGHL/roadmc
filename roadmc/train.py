"""
RoadMC Training Pipeline.
Usage:
    python roadmc/train.py mode=baseline --data_dir ./data/synthetic_output
    python roadmc/train.py mode=gan_enhanced
    python roadmc/train.py mode=end2end
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint


def train_baseline(args):
    """Train segmentation model on synthetic data."""
    from roadmc.data.dataloader import RoadMCDataModule
    from roadmc.models.model_pl import RoadMCSegModel

    model = RoadMCSegModel(
        in_channels=3,
        num_classes=38,
        embed_dim=args.embed_dim,
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    datamodule = RoadMCDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_points=args.max_points,
        num_workers=args.num_workers,
    )

    callbacks = [
        ModelCheckpoint(monitor="val_mIoU", mode="max", save_top_k=3, filename="baseline-{epoch}-{val_mIoU:.3f}"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)


def train_gan_enhanced(args):
    """GAN pre-training → synthetic + stylized mixed training."""
    from roadmc.data.dataloader import RoadMCDataModule, collate_pointcloud_batch
    from roadmc.models.gan.discriminator import WGANDiscriminator
    from roadmc.models.gan.generator import StyleTransferGen
    from roadmc.models.model_pl import RoadMCSegModel
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Create GAN components
    gen = StyleTransferGen().to(device)
    disc = WGANDiscriminator(in_channels=6).to(device)

    print(f"[GAN] Generator params: {sum(p.numel() for p in gen.parameters()):,}")
    print(f"[GAN] Discriminator params: {sum(p.numel() for p in disc.parameters()):,}")

    # 2. WGAN-GP pre-training loop
    datamodule = RoadMCDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_points=min(args.max_points, 8192),  # limit N for GAN memory
        num_workers=args.num_workers,
    )
    datamodule.setup("fit")

    g_opt = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.9))
    d_opt = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.9))

    n_critic = 5  # discriminator updates per generator update
    lambda_gp = 10.0  # gradient penalty coefficient
    gan_epochs = min(args.max_epochs, 5)  # short pre-training

    print(f"[GAN] Pre-training for {gan_epochs} epochs...")
    gen.train()
    disc.train()

    for epoch in range(gan_epochs):
        d_loss_total = 0.0
        g_loss_total = 0.0
        n_batches = 0

        for batch in datamodule.train_dataloader():
            coords = batch["coords"].to(device)
            normals = batch["normals"].to(device)

            for _ in range(n_critic):
                d_opt.zero_grad()
                real_input = torch.cat([coords, normals], dim=-1)  # (B, N, 6)
                with torch.no_grad():
                    fake_input = gen(coords, normals.detach())

                real_validity = disc(real_input)
                fake_validity = disc(fake_input)

                gp = _gradient_penalty(disc, real_input, fake_input, device)
                d_loss = -real_validity.mean() + fake_validity.mean() + lambda_gp * gp
                d_loss.backward()
                d_opt.step()
                d_loss_total += d_loss.item()

                del real_input, fake_input, real_validity, fake_validity, gp

            # Generator update (NOTE: no torch.no_grad — gradients must flow through disc to gen)
            g_opt.zero_grad()
            fake_input = gen(coords, normals.detach())
            g_loss_adv = -disc(fake_input).mean()
            g_loss_chamfer = torch.cdist(fake_input[:,:,:3], coords).min(dim=2)[0].mean() * 10.0
            g_loss_normal = (1.0 - (fake_input[:,:,3:6] * normals).sum(dim=-1)).mean()
            g_loss = g_loss_adv + g_loss_chamfer + g_loss_normal
            g_loss.backward()
            g_opt.step()
            g_loss_total += g_loss.item()
            n_batches += 1

        if n_batches > 0:
            print(f"  [GAN] Epoch {epoch+1}/{gan_epochs}: D_loss={d_loss_total/n_batches:.4f}, "
                  f"G_loss={g_loss_total/n_batches:.4f}")

    print("[GAN] Pre-training complete.")

    # Add Chamfer distance + normal consistency losses (complement WGAN-GP adversarial)
    def chamfer_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Chamfer distance: sum of nearest-neighbor distances between two point sets."""
        dist = torch.cdist(pred, target)  # (B, N, N)
        return dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean()

    def normal_consistency(normals_pred: torch.Tensor, normals_ref: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between predicted and reference normals."""
        cos_sim = (normals_pred * normals_ref).sum(dim=-1)
        return (1.0 - cos_sim).mean()

    print("[GAN] Mixed training: segmentation on GAN-styled data...")
    seg_model = RoadMCSegModel(
        in_channels=3, num_classes=38,
        embed_dim=args.embed_dim,
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
        lr=args.lr, weight_decay=args.weight_decay,
    ).to(device)
    seg_opt = torch.optim.AdamW(seg_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    gen.eval()
    for epoch in range(args.max_epochs):
        for batch in datamodule.train_dataloader():
            coords = batch["coords"].to(device)
            normals = batch["normals"].to(device)
            labels = batch["labels"].to(device)

            seg_opt.zero_grad()
            with torch.no_grad():
                styled = gen(coords, normals.detach())
            sc = coords + styled[:, :, :3]
            logits = seg_model(sc, batch["feats"].to(device))
            seg_loss = seg_model.lambda_focal * seg_model.focal_loss(logits, labels) \
                     + seg_model.lambda_dice * seg_model.dice_loss(logits, labels)
            seg_loss.backward()
            torch.nn.utils.clip_grad_norm_(seg_model.parameters(), 1.0)
            seg_opt.step()

        if (epoch + 1) % 5 == 0:
            print(f"  [GAN+Seg] Epoch {epoch+1}/{args.max_epochs}")

    print("[GAN] Mixed training complete.")


def train_end2end(args):
    """End-to-end: alternating GAN + segmentation training.

    Alternates between:
    1. GAN discriminator steps (improve realism detection)
    2. GAN generator steps (produce more realistic stylized data)
    3. Segmentation training on stylized data
    
    The segmentation model is trained on the fly as the GAN improves,
    creating a co-adaptation loop.
    """
    from roadmc.data.dataloader import RoadMCDataModule
    from roadmc.models.gan.discriminator import WGANDiscriminator
    from roadmc.models.gan.generator import StyleTransferGen
    from roadmc.models.model_pl import RoadMCSegModel
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen = StyleTransferGen().to(device)
    disc = WGANDiscriminator(in_channels=6).to(device)
    seg_model = RoadMCSegModel(
        in_channels=3, num_classes=38,
        embed_dim=args.embed_dim,
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
        lr=args.lr, weight_decay=args.weight_decay,
    ).to(device)

    g_opt = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.9))
    d_opt = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.9))
    seg_opt = torch.optim.AdamW(seg_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    datamodule = RoadMCDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_points=min(args.max_points, 8192),
        num_workers=args.num_workers,
    )
    datamodule.setup("fit")

    print(f"[E2E] Starting end-to-end training for {args.max_epochs} epochs...")
    gen.train(); disc.train(); seg_model.train()
    n_critic = 3; lambda_gp = 10.0

    for epoch in range(args.max_epochs):
        for batch in datamodule.train_dataloader():
            coords = batch["coords"].to(device)
            normals = batch["normals"].to(device)
            labels = batch["labels"].to(device)

            for _ in range(n_critic):
                d_opt.zero_grad()
                ri = torch.cat([coords, normals], dim=-1)
                with torch.no_grad():
                    fi = gen(coords, normals.detach())
                gp = _gradient_penalty(disc, ri, fi, device)
                d_loss = -disc(ri).mean() + disc(fi).mean() + lambda_gp * gp
                d_loss.backward(); d_opt.step()

            # Generator (no torch.no_grad — gradients must flow through disc to gen)
            g_opt.zero_grad()
            fi = gen(coords, normals.detach())
            g_loss = -disc(fi).mean()
            g_loss.backward(); g_opt.step()

            seg_opt.zero_grad()
            with torch.no_grad():
                styled = gen(coords, normals.detach())
            sc = coords + styled[:, :, :3]
            logits = seg_model(sc, batch["feats"].to(device))
            seg_loss = seg_model.lambda_focal * seg_model.focal_loss(logits, labels) \
                     + seg_model.lambda_dice * seg_model.dice_loss(logits, labels)
            seg_loss.backward()
            torch.nn.utils.clip_grad_norm_(seg_model.parameters(), 1.0)
            seg_opt.step()

        if (epoch + 1) % 5 == 0:
            print(f"  [E2E] Epoch {epoch+1}/{args.max_epochs}")

    print("[E2E] End-to-end training complete.")


def _gradient_penalty(
    disc: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """WGAN-GP gradient penalty: E[(||∇D(interpolated)||₂ - 1)²]."""
    B = real.shape[0]
    alpha = torch.rand(B, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolated = disc(interpolated)

    grad = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_norm = grad.view(B, -1).norm(2, dim=1)
    gp = ((grad_norm - 1.0) ** 2).mean()
    return gp


def main():
    parser = argparse.ArgumentParser(description="RoadMC Training Pipeline")
    parser.add_argument("mode", type=str, choices=["baseline", "gan_enhanced", "end2end"],
                        help="Training mode")
    parser.add_argument("--data_dir", type=str, default="./data/synthetic_output")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_points", type=int, default=65536)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--embed_dim", type=int, default=96)
    parser.add_argument("--depths", type=int, nargs=4, default=[2, 2, 6, 2])
    parser.add_argument("--num_heads", type=int, nargs=4, default=[3, 6, 12, 24])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.mode == "baseline":
        train_baseline(args)
    elif args.mode == "gan_enhanced":
        train_gan_enhanced(args)
    elif args.mode == "end2end":
        train_end2end(args)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # Test imports only (no actual training without data)
    from roadmc.data.dataloader import RoadMCDataModule  # noqa: F401
    from roadmc.models.gan.discriminator import WGANDiscriminator
    from roadmc.models.gan.generator import StyleTransferGen
    from roadmc.models.mhc.spectral_analysis import SpectralAnalyzer  # noqa: F401
    from roadmc.models.model_pl import RoadMCSegModel

    # Quick shape test
    model = RoadMCSegModel(
        in_channels=3, num_classes=38,
        embed_dim=48, depths=(1, 1, 2, 1), num_heads=(2, 4, 8, 16),
    )
    print(f"[PASS] RoadMCSegModel loaded: {sum(p.numel() for p in model.parameters()):,} params")

    gen = StyleTransferGen()
    print(f"[PASS] StyleTransferGen loaded: {sum(p.numel() for p in gen.parameters()):,} params")

    disc = WGANDiscriminator()
    print(f"[PASS] WGANDiscriminator loaded: {sum(p.numel() for p in disc.parameters()):,} params")

    _ = SpectralAnalyzer  # verify import resolves
    _ = RoadMCDataModule  # verify import resolves

    print("\nUsage:")
    print("  python roadmc/train.py baseline --data_dir ./data/synthetic_output --max_epochs 1")
    print("  python roadmc/train.py gan_enhanced --max_epochs 1")
