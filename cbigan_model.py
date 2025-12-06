import os
from pathlib import Path

import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.io import fits
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt


class CBiGANPipeline:


    class ImageTransforms:
        """Transforms for astronomical FITS images."""

        @staticmethod
        def normalize_to_range(image, target_range=(-1, 1)):
            """Normalize an image to the specified range."""
            min_val, max_val = image.min(), image.max()
            if max_val > min_val:
                normalized = (image - min_val) / (max_val - min_val)
                normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
            else:
                normalized = np.zeros_like(image)
            return normalized.astype(np.float32)

        @staticmethod
        def resize_image(image, size=(128, 128)):
            """Resize an image using interpolation."""
            from skimage.transform import resize

            return resize(image, size, anti_aliasing=True, preserve_range=True)

    class AstronomicalFitsDataset(Dataset):
        """Dataset for astronomical FITS files."""

        def __init__(self, root_dir, max_files=None):
            self.root_dir = Path(root_dir)
            self.data = []

            fits_files = list(self.root_dir.glob("*.fits"))
            if max_files:
                fits_files = fits_files[:max_files]

            print(f"Loading {len(fits_files)} FITS files...")

            for fits_file in tqdm(fits_files):
                try:
                    with fits.open(fits_file) as hdul:
                        for hdu in hdul:
                            if hdu.data is not None:
                                image = hdu.data.astype(np.float32)
                                image = CBiGANPipeline.ImageTransforms.resize_image(image, (128, 128))
                                image = CBiGANPipeline.ImageTransforms.normalize_to_range(image)
                                tensor = torch.from_numpy(image).unsqueeze(0)
                                self.data.append(tensor)
                except Exception as e:
                    print(f"Error loading {fits_file}: {e}")
                    continue

            print(f"Dataset loaded: {len(self.data)} images")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    class Generator(nn.Module):
        """Generator: latent vector to image."""

        def __init__(self, latent_dim=64, img_channels=1, base_filters=32):
            super().__init__()
            self.fc = nn.Linear(latent_dim, base_filters * 8 * 8 * 8)
            self.conv_blocks = nn.Sequential(
                nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 4, 2, 1),
                nn.BatchNorm2d(base_filters * 4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 4, 2, 1),
                nn.BatchNorm2d(base_filters * 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base_filters * 2, base_filters, 4, 2, 1),
                nn.BatchNorm2d(base_filters),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base_filters, img_channels, 4, 2, 1),
                nn.Tanh(),
            )

        def forward(self, z):
            x = self.fc(z)
            x = x.view(x.size(0), -1, 8, 8)
            x = self.conv_blocks(x)
            return x

    class Encoder(nn.Module):
        """Encoder: image to latent vector."""

        def __init__(self, latent_dim=64, img_channels=1, base_filters=32):
            super().__init__()
            self.conv_blocks = nn.Sequential(
                nn.Conv2d(img_channels, base_filters, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_filters, base_filters * 2, 4, 2, 1),
                nn.BatchNorm2d(base_filters * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_filters * 2, base_filters * 4, 4, 2, 1),
                nn.BatchNorm2d(base_filters * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_filters * 4, base_filters * 8, 4, 2, 1),
                nn.BatchNorm2d(base_filters * 8),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.fc = nn.Linear(base_filters * 8 * 8 * 8, latent_dim)

        def forward(self, x):
            x = self.conv_blocks(x)
            x = x.view(x.size(0), -1)
            z = self.fc(x)
            return z

    class Discriminator(nn.Module):
        """Discriminator: joint image-latent pair classifier."""

        def __init__(self, latent_dim=64, img_channels=1, base_filters=32, dropout_rate=0.3):
            super().__init__()
            self.img_encoder = nn.Sequential(
                nn.Conv2d(img_channels, base_filters, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_filters, base_filters * 2, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_filters * 2, base_filters * 4, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_filters * 4, base_filters * 8, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(),
                nn.Linear(base_filters * 8 * 8 * 8, 256),
            )
            self.z_encoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, 128),
            )
            self.classifier = nn.Sequential(
                nn.Linear(256 + 128, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 1),
            )

        def forward(self, x, z):
            img_feat = self.img_encoder(x)
            z_feat = self.z_encoder(z)
            combined = torch.cat([img_feat, z_feat], dim=1)
            out = self.classifier(combined)
            return out

    class AnomalyMetrics:
        """Metrics for anomaly detection."""

        def __init__(self, device):
            self.device = device

        def reconstruction_error(self, original, reconstructed):
            """Mean squared reconstruction error."""
            return F.mse_loss(original, reconstructed, reduction="none").mean(dim=[1, 2, 3])

        def ssim_score(self, original, reconstructed):
            """Structural similarity index between images."""
            scores = []
            for i in range(original.shape[0]):
                orig = original[i].squeeze().cpu().numpy()
                recon = reconstructed[i].squeeze().cpu().numpy()
                orig = (orig + 1) / 2
                recon = (recon + 1) / 2
                score = ssim(orig, recon, data_range=1.0)
                scores.append(score)
            return torch.tensor(scores, device=self.device)

        def discriminator_error(self, discriminator, image, encoded_z):
            """Discriminator-based anomaly metric."""
            with torch.no_grad():
                d_score = torch.sigmoid(discriminator(image, encoded_z))
            return 1 - d_score.squeeze()

    def __init__(self, device=None, output_dir_images="reconstructions", output_dir_grids="image_samples"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.output_dir_images = Path(output_dir_images)
        self.output_dir_grids = Path(output_dir_grids)
        self.create_directory(self.output_dir_images)
        self.create_directory(self.output_dir_grids)

    @staticmethod
    def create_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_image(self, image, path, filename):
        """Save an image to the specified path."""
        file_path = Path(path) / filename
        print(f"Saving image to: {file_path}")
        plt.imsave(file_path, image, cmap="gray")
        print(f"Image saved: {filename}")

    def create_dataloaders(self, dataset, train_ratio=0.8, batch_size=16):
        """Create train and validation dataloaders."""
        train_size = int(len(dataset) * train_ratio)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def train_cbigan_with_params(
        self,
        trial_params,
        dataset_path,
        max_files=20,
        max_epochs=50,
        early_stopping_patience=10,
        device=None,
        verbose=False,
    ):
        """
        Train the CBiGAN model with the provided hyperparameters.

        Args:
            trial_params (dict): Hyperparameter configuration.
            dataset_path (str): Dataset root path.
            max_files (int): Maximum number of files to load.
            max_epochs (int): Maximum number of epochs.
            early_stopping_patience (int): Patience for early stopping.
            device (str or torch.device): Target device for training.
            verbose (bool): Whether to print progress logs.

        Returns:
            float: Validation metric to minimize (lower is better).
        """

        latent_dim = trial_params["latent_dim"]
        learning_rate = trial_params["learning_rate"]
        batch_size = trial_params["batch_size"]
        d_lr_factor = trial_params["d_lr_factor"]
        recon_weight = trial_params["recon_weight"]
        label_smoothing = trial_params["label_smoothing"]
        base_filters = trial_params["base_filters"]
        dropout_rate = trial_params["dropout_rate"]

        device = device or self.device

        try:
            dataset = CBiGANPipeline.AstronomicalFitsDataset(dataset_path, max_files=max_files)
            if len(dataset) == 0:
                return float("inf")

            train_loader, val_loader = self.create_dataloaders(dataset, batch_size=batch_size)

            G = CBiGANPipeline.Generator(latent_dim, base_filters=base_filters).to(device)
            E = CBiGANPipeline.Encoder(latent_dim, base_filters=base_filters).to(device)
            D = CBiGANPipeline.Discriminator(latent_dim, base_filters=base_filters, dropout_rate=dropout_rate).to(device)

            opt_G = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
            opt_E = torch.optim.Adam(E.parameters(), lr=learning_rate, betas=(0.5, 0.999))
            opt_D = torch.optim.Adam(D.parameters(), lr=learning_rate * d_lr_factor, betas=(0.5, 0.999))

            criterion = nn.BCEWithLogitsLoss()
            metrics = CBiGANPipeline.AnomalyMetrics(device)

            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(max_epochs):
                G.train()
                E.train()
                D.train()

                train_d_loss = 0
                train_ge_loss = 0
                train_recon_error = 0

                for batch_idx, real_images in enumerate(train_loader):
                    real_images = real_images.to(device)
                    batch_size_actual = real_images.size(0)

                    if batch_idx % 2 == 0:
                        opt_D.zero_grad()
                        real_z = E(real_images)
                        d_real = D(real_images, real_z.detach())
                        fake_z = torch.randn(batch_size_actual, latent_dim, device=device)
                        fake_images = G(fake_z)
                        d_fake = D(fake_images.detach(), fake_z)

                        real_labels = torch.ones_like(d_real) * (1.0 - label_smoothing)
                        fake_labels = torch.zeros_like(d_fake) + label_smoothing
                        d_loss = (criterion(d_real, real_labels) + criterion(d_fake, fake_labels)) / 2
                        d_loss.backward()
                        opt_D.step()
                        train_d_loss += d_loss.item()

                    opt_G.zero_grad()
                    opt_E.zero_grad()

                    real_z_ge = E(real_images)
                    fake_z_ge = torch.randn(batch_size_actual, latent_dim, device=device)
                    fake_images_ge = G(fake_z_ge)
                    d_fake_ge = D(fake_images_ge, fake_z_ge)
                    d_real_ge = D(real_images, real_z_ge)

                    adv_loss = (criterion(d_fake_ge, torch.ones_like(d_fake_ge)) + criterion(d_real_ge, torch.zeros_like(d_real_ge))) / 2
                    reconstructed = G(real_z_ge)
                    recon_loss = F.mse_loss(real_images, reconstructed) * recon_weight
                    ge_loss = adv_loss + recon_loss

                    ge_loss.backward()
                    opt_G.step()
                    opt_E.step()

                    train_ge_loss += ge_loss.item()
                    train_recon_error += recon_loss.item()

                G.eval()
                E.eval()
                D.eval()

                val_recon_errors = []
                val_ssim_scores = []

                with torch.no_grad():
                    for val_images in val_loader:
                        val_images = val_images.to(device)
                        z = E(val_images)
                        reconstructed = G(z)
                        recon_error = metrics.reconstruction_error(val_images, reconstructed)
                        ssim_scores = metrics.ssim_score(val_images, reconstructed)

                        val_recon_errors.extend(recon_error.cpu().numpy())
                        val_ssim_scores.extend(ssim_scores.cpu().numpy())

                avg_recon_error = np.mean(val_recon_errors)
                avg_ssim = np.mean(val_ssim_scores)
                val_loss = avg_recon_error + (1 - avg_ssim)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{max_epochs} | Val Loss: {val_loss:.4f} | "
                        f"Recon: {avg_recon_error:.4f} | SSIM: {avg_ssim:.4f}"
                    )

            return best_val_loss

        except Exception as e:
            print(f"Error during training: {e}")
            return float("inf")

    def objective_function(self, trial, dataset_path, max_files=20):
        """Optuna objective function for hyperparameter optimization."""
        trial_params = {
            "latent_dim": trial.suggest_int("latent_dim", 32, 128, step=16),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "d_lr_factor": trial.suggest_float("d_lr_factor", 0.1, 1.0),
            "recon_weight": trial.suggest_float("recon_weight", 1.0, 50.0),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.3),
            "base_filters": trial.suggest_categorical("base_filters", [16, 32, 64]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        }

        val_loss = self.train_cbigan_with_params(
            trial_params=trial_params,
            dataset_path=dataset_path,
            max_files=max_files,
            max_epochs=50,
            early_stopping_patience=10,
            device=self.device,
            verbose=False,
        )

        return val_loss

    def example_single_training(self, dataset_path="./dataset01/", max_files=20):
        """Run a single training example with predefined parameters."""
        example_params = {
            "latent_dim": 64,
            "learning_rate": 0.0002,
            "batch_size": 16,
            "d_lr_factor": 0.5,
            "recon_weight": 10.0,
            "label_smoothing": 0.1,
            "base_filters": 32,
            "dropout_rate": 0.3,
        }

        val_loss = self.train_cbigan_with_params(
            trial_params=example_params,
            dataset_path=dataset_path,
            max_files=max_files,
            max_epochs=100,
            early_stopping_patience=15,
            device=self.device,
            verbose=True,
        )

        print(f"Final validation loss: {val_loss:.4f}")
        return val_loss


if __name__ == "__main__":
    pipeline = CBiGANPipeline()
    # Update dataset_path with the local dataset location before running.
    pipeline.example_single_training(dataset_path="./dataset01/")
