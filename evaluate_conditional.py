"""
Evaluate Conditional Diffusion Model (2-label: Ωₘ, σ₈)

Usage:
    python evaluate_conditional.py --checkpoint outputs_conditional_YYYYMMDD_HHMMSS
"""

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion_conditional import GaussianDiffusion, ConditionalDiffusionModel
from unet_conditional import ConditionalUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate conditional 2-label diffusion model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (e.g. outputs_conditional_*/checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--training_args",
        type=str,
        default=None,
        help="Path to args.txt from training (auto-detected if not provided)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/params_2",
        help="Directory containing the CAMELS LH dataset (default matches repo structure)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to use for real images",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of examples to show in the comparison grid",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_outputs",
        help="Where to save plots and results",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps",
    )
    return parser.parse_args()


def load_training_config(path: str) -> Dict:
    """Load training args.txt and parse values correctly."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Training args file not found: {path}")

    config = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            # Parse lists and numbers safely
            if value.startswith("[") and value.endswith("]"):
                try:
                    config[key] = ast.literal_eval(value)
                except:
                    config[key] = value
            elif value.isdigit():
                config[key] = int(value)
            elif value.replace(".", "", 1).replace("e-", "", 1).replace("e", "", 1).isdigit():
                config[key] = float(value)
            else:
                config[key] = value

    return config


def load_label_stats(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load mean and std from training labels (used for normalization)."""
    labels_path = data_dir / "train_labels_LH_2.npy"
    if not labels_path.exists():
        raise FileNotFoundError(f"Label statistics file not found: {labels_path}")
    labels = np.load(labels_path)
    return labels.mean(axis=0), labels.std(axis=0)


def load_split(data_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load images and labels for a given split."""
    image_path = data_dir / f"{split}_LH.npy"
    label_path = data_dir / f"{split}_labels_LH_2.npy"

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    images = np.load(image_path).astype(np.float32)
    labels = np.load(label_path).astype(np.float32)
    return images, labels


def build_model(config: Dict, device: torch.device) -> ConditionalDiffusionModel:
    """Rebuild the exact same model architecture used during training."""
    unet = ConditionalUNet(
        in_channels=1,
        out_channels=1,
        label_dim=int(config.get("label_dim", 2)),
        base_channels=int(config.get("base_channels", 64)),
        channel_multipliers=config.get("channel_multipliers", [1, 2, 4, 8]),
        attention_levels=config.get("attention_levels", [2, 3]),
        dropout=float(config.get("dropout", 0.1)),
    )

    diffusion = GaussianDiffusion(
        timesteps=int(config.get("timesteps", 1500)),
        beta_start=float(config.get("beta_start", 1e-4)),
        beta_end=float(config.get("beta_end", 0.02)),
        schedule_type=config.get("schedule_type", "linear"),
    ).to(device)

    return ConditionalDiffusionModel(unet, diffusion).to(device)


def load_checkpoint(model: ConditionalDiffusionModel, checkpoint_path: str, device: torch.device):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")


# ------------------------------------------------------------------
# The rest of the functions (PowerSpectrum, PDF, plotting, etc.) remain unchanged
# ------------------------------------------------------------------

def PowerSpectrum(box: np.ndarray, N: int, dl: float) -> Tuple[np.ndarray, np.ndarray]:
    FT_box = np.fft.fftn(box, norm="ortho")
    k = 2 * np.pi * np.fft.fftfreq(N, dl)
    pk = np.zeros(N)
    count = np.zeros(N)
    dk_val = 2 * np.pi / (N * dl)

    for i in range(N):
        for j in range(N):
            kbar = np.sqrt(k[i]**2 + k[j]**2)
            t = int(round(kbar / dk_val))
            if t < N:
                count[t] += 1.0
                pk[t] += FT_box[i, j] * np.conj(FT_box[i, j])

    pk /= np.where(count == 0, 1, count)
    pk *= dl**2
    dk = np.arange(N) * dk_val
    return dk, pk.real


def calculate_pdf_batch(images: np.ndarray, log_nhi_min=14.0, log_nhi_max=22.0, n_bins=100):
    images_01 = np.clip(images, 0.0, 1.0)
    log_nhi_bins = np.linspace(log_nhi_min, log_nhi_max, n_bins)
    bin_centers = 0.5 * (log_nhi_bins[:-1] + log_nhi_bins[1:])

    pdfs = []
    for img in images_01:
        log_nhi_values = log_nhi_min + (log_nhi_max - log_nhi_min) * img.reshape(-1)
        hist, _ = np.histogram(log_nhi_values, bins=log_nhi_bins, density=True)
        pdfs.append(hist)

    pdf_array = np.stack(pdfs)
    return bin_centers, pdf_array.mean(axis=0), pdf_array.std(axis=0)


def calculate_power_spectrum_batch(images: np.ndarray, box_size: float = 25.0):
    N = images.shape[-1]
    dl = box_size / N
    power_spectra = [PowerSpectrum(img, N=N, dl=dl)[1] for img in images]
    power_array = np.stack(power_spectra)
    return PowerSpectrum(images[0], N=N, dl=dl)[0], power_array.mean(axis=0), power_array.std(axis=0)


def prepare_labels_for_model(labels: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    normalized = (labels - mean) / std
    return torch.from_numpy(normalized).float()


def from_model_output(samples: torch.Tensor) -> np.ndarray:
    arrays = samples.cpu().numpy()
    return np.clip((arrays + 1.0) / 2.0, 0.0, 1.0)[:, 0, :, :]


def plot_image_grid(generated, real, labels, output_path: Path, num_samples=8):
    num = min(num_samples, generated.shape[0])
    fig, axes = plt.subplots(num, 2, figsize=(6, 3 * num))
    if num == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num):
        label_str = ", ".join(f"{v:.3f}" for v in labels[i])
        axes[i, 0].imshow(generated[i], cmap="magma", origin="lower")
        axes[i, 0].set_title(f"Generated\n{label_str}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(real[i], cmap="magma", origin="lower")
        axes[i, 1].set_title("Real")
        axes[i, 1].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mean_std(x, mean_real, std_real, mean_gen, std_gen, xlabel, ylabel, title, output_path: Path, yscale="linear"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, mean_real, label="Real mean", color="tab:blue", linewidth=2)
    ax.plot(x, mean_gen, label="Generated mean", color="tab:orange", linewidth=2)

    ax.fill_between(x, mean_real - std_real, mean_real + std_real, color="tab:blue", alpha=0.15, label="Real ±1σ")
    ax.fill_between(x, mean_real - 3*std_real, mean_real + 3*std_real, color="tab:blue", alpha=0.05)

    ax.fill_between(x, mean_gen - std_gen, mean_gen + std_gen, color="tab:orange", alpha=0.15, label="Generated ±1σ")
    ax.fill_between(x, mean_gen - 3*std_gen, mean_gen + 3*std_gen, color="tab:orange", alpha=0.05)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale(yscale)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training config
    if args.training_args is None:
        # Auto-find the most recent args.txt (optional enhancement)
        possible = list(Path(".").glob("outputs_conditional_*/args.txt"))
        if possible:
            args.training_args = str(max(possible, key=os.path.getctime))
            print(f"Auto-detected training args: {args.training_args}")
        else:
            raise FileNotFoundError("Please provide --training_args path to your training args.txt")

    config = load_training_config(args.training_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    load_checkpoint(model, args.checkpoint, device)

    # Load data
    data_dir = Path(args.data_dir)
    images_split, labels_split = load_split(data_dir, args.split)
    label_mean, label_std = load_label_stats(data_dir)

    # Select random samples
    num_select = min(100, len(images_split))
    indices = np.random.choice(len(images_split), num_select, replace=False)

    real_images = images_split[indices]
    original_labels = labels_split[indices]

    # Generate samples in batches
    batch_size = min(8, num_select)
    generated_list = []

    print(f"Generating {num_select} samples (batch size = {batch_size})...")
    for i in range(0, num_select, batch_size):
        batch_labels = original_labels[i:i+batch_size]
        batch_labels_tensor = prepare_labels_for_model(batch_labels, label_mean, label_std).to(device)

        with torch.no_grad():
            batch_gen = model.sample(
                labels=batch_labels_tensor,
                channels=1,
                height=real_images.shape[-2],
                width=real_images.shape[-1],
                device=device,
                progress=False,
                use_ddim=True,
                ddim_steps=args.ddim_steps,
            )
        generated_list.append(from_model_output(batch_gen))
        print(f"  Batch {i//batch_size + 1}/{(num_select+batch_size-1)//batch_size} done")

    generated_images = np.concatenate(generated_list, axis=0)

    # Plots
    plot_image_grid(generated_images, real_images, original_labels,
                    output_dir / "real_vs_generated.png", num_samples=args.num_samples)

    # PDF
    bin_centers, mean_pdf_real, std_pdf_real = calculate_pdf_batch(real_images)
    _, mean_pdf_gen, std_pdf_gen = calculate_pdf_batch(generated_images)
    plot_mean_std(bin_centers, mean_pdf_real, std_pdf_real, mean_pdf_gen, std_pdf_gen,
                  "log N_HI [cm⁻²]", "PDF", "Column Density PDF", output_dir / "pdf_mean_std.png")

    # Power Spectrum
    dk, mean_pk_real, std_pk_real = calculate_power_spectrum_batch(real_images)
    _, mean_pk_gen, std_pk_gen = calculate_power_spectrum_batch(generated_images)
    plot_mean_std(dk, mean_pk_real, std_pk_real, mean_pk_gen, std_pk_gen,
                  "k [h/Mpc]", "P(k)", "Power Spectrum", output_dir / "power_spectrum_mean_std.png", yscale="log")

    # Save numerical results
    np.savez(
        output_dir / "evaluation_data.npz",
        indices=indices,
        labels_original=original_labels,
        bin_centers=bin_centers,
        mean_pdf_real=mean_pdf_real, std_pdf_real=std_pdf_real,
        mean_pdf_gen=mean_pdf_gen, std_pdf_gen=std_pdf_gen,
        dk=dk,
        mean_pk_real=mean_pk_real, std_pk_real=std_pk_real,
        mean_pk_gen=mean_pk_gen, std_pk_gen=std_pk_gen,
    )

    print(f"\nEvaluation complete!")
    print(f"   Plots saved to: {output_dir}")
    print(f"   Numerical data saved to: {output_dir}/evaluation_data.npz")


if __name__ == "__main__":
    main()
