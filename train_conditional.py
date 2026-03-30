"""
Training Script for Conditional Diffusion Model
Trains diffusion model conditioned on cosmological parameters (Omega_m, sigma_8)
"""

import torch
import torch.optim as optim
import numpy as np
import os
import argparse
import random
from tqdm import tqdm
import time

from unet_conditional import ConditionalUNet
from diffusion_conditional import GaussianDiffusion, ConditionalDiffusionModel
from dataset_conditional import get_conditional_dataloaders
import matplotlib.pyplot as plt

# Weights & Biases (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        self.backup = {name: param.data.clone() for name, param in self.model.named_parameters() if param.requires_grad}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def train_epoch(model, dataloader, optimizer, device, epoch, ema=None, use_wandb=False):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = model.get_loss(images, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if ema is not None:
            ema.update()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if use_wandb and batch_idx % 10 == 0:
            wandb.log({'batch_loss': loss.item(), 'epoch': epoch, 'batch': epoch * len(dataloader) + batch_idx})

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)
            loss = model.get_loss(images, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, ema, epoch, loss, save_dir, is_best=False, last_improvement_epoch=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if ema is not None:
        checkpoint['ema_shadow'] = ema.shadow
    if last_improvement_epoch is not None:
        checkpoint['last_improvement_epoch'] = last_improvement_epoch

    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_latest.pt'))
    if is_best:
        torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
        print(f"Saved best model at epoch {epoch+1}")

    if (epoch + 1) % 20 == 0:
        torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))

    print(f"Saved checkpoint at epoch {epoch+1}")


def sample_images(model, diffusion, device, save_path, test_labels, n_samples=8, epoch=0, use_ddim=True, ddim_steps=50, use_wandb=False):
    model.eval()
    labels = test_labels[:n_samples].to(device)

    with torch.no_grad():
        samples = diffusion.sample(
            model, labels=labels, channels=1, height=256, width=256,
            device=device, progress=True, use_ddim=use_ddim,
            ddim_steps=ddim_steps, eta=0.0
        )

    param_names = ['Ωₘ', 'σ₈']
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            img = samples[i, 0].cpu().numpy()
            label_str = f"{param_names[0]}={labels[i,0]:.2f}, {param_names[1]}={labels[i,1]:.2f}"
            ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
            ax.set_title(label_str, fontsize=10)
            ax.axis('off')

    plt.suptitle(f'Generated Samples - Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if use_wandb:
        wandb.log({'generated_samples': wandb.Image(save_path), 'epoch': epoch})
    plt.close()
    print(f"Saved samples to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Conditional Diffusion Model')
    # Model
    parser.add_argument('--label_dim', type=int, default=2)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--channel_multipliers', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--attention_levels', type=int, nargs='+', default=[2, 3])
    parser.add_argument('--dropout', type=float, default=0.1)
    # Diffusion
    parser.add_argument('--timesteps', type=int, default=1500)
    parser.add_argument('--beta_start', type=float, default=1e-4)
    parser.add_argument('--beta_end', type=float, default=0.02)
    parser.add_argument('--schedule_type', type=str, default='linear')
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--early_stop_patience', type=int, default=30)
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/params_2',
                        help='Data directory (relative to repo root)')
    parser.add_argument('--normalize_labels', action='store_true', default=True)
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs_conditional')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--sample_every', type=int, default=10)
    parser.add_argument('--use_ddim', action='store_true', default=True)
    parser.add_argument('--ddim_steps', type=int, default=50)
    # WandB
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='ddpm_cosmology')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_run_name', type=str, default='')

    args = parser.parse_args()

    # Reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # WandB
    use_wandb = args.use_wandb and WANDB_AVAILABLE
    if use_wandb:
        run_name = args.wandb_run_name or f"conditional_diffusion_{time.strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, name=run_name, config=vars(args))
        print(f"W&B run: {run_name}")

    # Directories
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_conditional_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize_labels=args.normalize_labels
    )
    _, test_labels = next(iter(test_loader))

    # Model
    print("\nCreating model...")
    unet = ConditionalUNet(
        in_channels=1, out_channels=1, label_dim=args.label_dim,
        base_channels=args.base_channels,
        channel_multipliers=args.channel_multipliers,
        attention_levels=args.attention_levels,
        dropout=args.dropout
    )
    diffusion = GaussianDiffusion(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        schedule_type=args.schedule_type
    ).to(device)

    model = ConditionalDiffusionModel(unet, diffusion).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ema = EMA(model, decay=args.ema_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    last_improvement_epoch = -1
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'ema_shadow' in checkpoint:
            ema.shadow = checkpoint['ema_shadow']
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('loss', float('inf'))
        last_improvement_epoch = checkpoint.get('last_improvement_epoch', -1)

    # Training
    print("\nStarting training...")
    losses = {'train': [], 'val': []}

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, ema, use_wandb)
        val_loss = validate(model, val_loader, device)
        losses['train'].append(train_loss)
        losses['val'].append(val_loss)
        scheduler.step()

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

        print(f"\nEpoch {epoch+1}/{args.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6e}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            last_improvement_epoch = epoch

        save_checkpoint(model, optimizer, ema, epoch, val_loss,
                        os.path.join(output_dir, 'checkpoints'),
                        is_best=is_best,
                        last_improvement_epoch=last_improvement_epoch)

        # Early stopping
        if epoch - last_improvement_epoch >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Samples
        if (epoch + 1) % args.sample_every == 0:
            sample_path = os.path.join(output_dir, 'samples', f'samples_epoch_{epoch+1}.png')
            sample_images(model, diffusion, device, sample_path, test_labels,
                          epoch=epoch+1, use_ddim=args.use_ddim,
                          ddim_steps=args.ddim_steps, use_wandb=use_wandb)

        # Loss plot
        if (epoch + 1) % 5 == 0:
            plt.figure(figsize=(10, 5))
            plt.plot(losses['train'], label='Train Loss')
            plt.plot(losses['val'], label='Val Loss')
            plt.yscale('log')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'losses.png'), dpi=150)
            plt.close()

    print(f"\nTraining completed! Best val loss: {best_val_loss:.6f}")
    print(f"Results saved to: {output_dir}")
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
