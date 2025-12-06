import glob
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from astropy.io import fits
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize, rotate
from torch.utils.data import DataLoader, Dataset


class CBiGANDataAugPipeline:
    class RandomRotation:
        """Apply a random rotation. Galaxies can appear at any orientation."""

        def __init__(self, max_angle: float = 180) -> None:
            self.max_angle = max_angle

        def __call__(self, image: np.ndarray) -> np.ndarray:
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            return rotate(image, angle, mode="reflect", preserve_range=True)

    class RandomFlip:
        """Apply a random horizontal or vertical flip."""

        def __init__(self, horizontal_prob: float = 0.5, vertical_prob: float = 0.5) -> None:
            self.horizontal_prob = horizontal_prob
            self.vertical_prob = vertical_prob

        def __call__(self, image: np.ndarray) -> np.ndarray:
            if np.random.random() < self.horizontal_prob:
                image = np.fliplr(image)
            if np.random.random() < self.vertical_prob:
                image = np.flipud(image)
            return image

    class AddGaussianNoise:
        """Add Gaussian noise to simulate observational variability."""

        def __init__(self, mean: float = 0, std_range: tuple[float, float] = (0.01, 0.05)) -> None:
            self.mean = mean
            self.std_range = std_range

        def __call__(self, image: np.ndarray) -> np.ndarray:
            std = np.random.uniform(self.std_range[0], self.std_range[1])
            noise = np.random.normal(self.mean, std, image.shape)
            return image + noise

    class RandomBrightness:
        """Adjust brightness to simulate different exposure times."""

        def __init__(self, brightness_range: tuple[float, float] = (0.7, 1.3)) -> None:
            self.brightness_range = brightness_range

        def __call__(self, image: np.ndarray) -> np.ndarray:
            factor = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
            return np.clip(image * factor, 0, np.max(image))

    class Rescale:
        """Rescale the image to a target size."""

        def __init__(self, output_size: tuple[int, int] = (128, 128)) -> None:
            self.output_size = output_size

        def __call__(self, image: np.ndarray) -> np.ndarray:
            h, w = image.shape[:2]
            new_h, new_w = self.output_size

            if (h, w) == (new_h, new_w):
                return image

            resized = resize(image, (new_h, new_w), anti_aliasing=True)
            return resized

    class Normalize:
        """Normalize the image by subtracting the mean and dividing by the standard deviation."""

        def __call__(self, image: np.ndarray) -> np.ndarray:
            mean = np.mean(image)
            std = np.std(image)
            if std == 0:
                return image - mean
            return (image - mean) / std

    class ToTensor:
        """Convert image arrays to PyTorch tensors (C x H x W)."""

        def __init__(self, device: torch.device | str = "cpu") -> None:
            self.device = torch.device(device)

        def __call__(self, image: np.ndarray) -> torch.Tensor:
            tensor = torch.from_numpy(image).float()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 3:
                tensor = tensor.permute(2, 0, 1)
            else:
                raise ValueError(f"Unsupported dimensions: {tensor.shape}")

            return tensor.to(self.device)

    class FitsDataset(Dataset):
        """Dataset capable of loading FITS files for training or evaluation."""

        def __init__(self, root_dir: str, transform: Optional[callable] = None) -> None:
            if not os.path.exists(root_dir):
                raise FileNotFoundError(f"Directory {root_dir} not found.")

            self.root_dir = root_dir
            self.transform = transform
            self.data: list[np.ndarray | torch.Tensor] = []

            fits_files = glob.glob(os.path.join(root_dir, "*.fits"))

            if not fits_files:
                raise FileNotFoundError(f"No FITS files found in {root_dir}.")

            print(f"Starting to load {len(fits_files)} FITS files...")

            count = 0
            for file_fit in fits_files:
                try:
                    hdul = fits.open(file_fit)

                    for hdu in hdul:
                        if hdu.data is not None:
                            data = hdu.data

                            if hasattr(data, "dtype") and not data.dtype.isnative:
                                data = data.astype(data.dtype.newbyteorder("="))

                            data = np.asarray(data, dtype=np.float32)

                            if self.transform:
                                data = self.transform(data)
                            self.data.append(data)

                    hdul.close()
                    count += 1

                    if count % 5 == 0 or count == len(fits_files):
                        percentage = (count / len(fits_files)) * 100
                        print(f"Loaded {count} FITS files of {len(fits_files)} ({percentage:.2f}%).")

                except Exception as exc:
                    print(f"Error reading file {file_fit}: {exc}")
                    continue

            self._print_memory_info()

        def _print_memory_info(self) -> None:
            if not self.data:
                print("Dataset is empty.")
                return

            total_size_bytes = 0
            for item in self.data:
                if isinstance(item, torch.Tensor):
                    total_size_bytes += item.element_size() * item.nelement()
                elif isinstance(item, np.ndarray):
                    total_size_bytes += item.nbytes

            size_mb = total_size_bytes / (1024 * 1024)
            size_gb = size_mb / 1024

            print("\n=== DATASET INFORMATION ===")
            print("Dataset loaded in RAM.")
            print(f"Number of images: {len(self.data)}")
            print(f"Total memory size: {size_mb:.2f} MB ({size_gb:.3f} GB)")
            print(f"Average size per image: {size_mb / len(self.data):.2f} MB")

            first_item = self.data[0]
            if isinstance(first_item, torch.Tensor):
                print(f"Format: PyTorch Tensor {first_item.shape}, dtype: {first_item.dtype}")
            elif isinstance(first_item, np.ndarray):
                print(f"Format: NumPy Array {first_item.shape}, dtype: {first_item.dtype}")
            else:
                print(f"Unknown format: {type(first_item)}")

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> np.ndarray | torch.Tensor:
            return self.data[idx]

    class Generator(nn.Module):
        """Generator: z -> x (latent vector to image)."""

        def __init__(self, latent_dim: int = 64, img_channels: int = 1) -> None:
            super().__init__()

            self.fc = nn.Linear(latent_dim, 256 * 8 * 8)

            self.conv_blocks = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
                nn.Tanh(),
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            x = self.fc(z)
            x = x.view(x.size(0), 256, 8, 8)
            x = self.conv_blocks(x)
            return x

    class Encoder(nn.Module):
        """Encoder: x -> z (image to latent vector)."""

        def __init__(self, latent_dim: int = 64, img_channels: int = 1) -> None:
            super().__init__()

            self.conv_blocks = nn.Sequential(
                nn.Conv2d(img_channels, 32, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
            )

            self.fc = nn.Linear(256 * 8 * 8, latent_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv_blocks(x)
            x = x.view(x.size(0), -1)
            z = self.fc(x)
            return z

    class Discriminator(nn.Module):
        """Discriminator: (x, z) -> real/fake."""

        def __init__(self, latent_dim: int = 64, img_channels: int = 1) -> None:
            super().__init__()

            self.img_encoder = nn.Sequential(
                nn.Conv2d(img_channels, 32, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(),
                nn.Linear(256 * 8 * 8, 256),
            )

            self.z_encoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, 128),
            )

            self.classifier = nn.Sequential(
                nn.Linear(256 + 128, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
            )

        def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            img_feat = self.img_encoder(x)
            z_feat = self.z_encoder(z)
            combined = torch.cat([img_feat, z_feat], dim=1)
            out = self.classifier(combined)
            return out

    class AnomalyMetrics:
        """Metrics for anomaly detection."""

        def __init__(self, device: torch.device | str) -> None:
            self.device = torch.device(device)

        def reconstruction_error(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
            return F.mse_loss(original, reconstructed, reduction="none").mean(dim=[1, 2, 3])

        def ssim_score(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
            scores: list[float] = []
            for i in range(original.shape[0]):
                orig = original[i].squeeze().cpu().numpy()
                recon = reconstructed[i].squeeze().cpu().numpy()

                orig = (orig + 1) / 2
                recon = (recon + 1) / 2

                score = ssim(orig, recon, data_range=1.0)
                scores.append(score)

            return torch.tensor(scores, device=self.device)

        def discriminator_error(self, discriminator: nn.Module, image: torch.Tensor, encoded_z: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                d_score = torch.sigmoid(discriminator(image, encoded_z))
            return 1 - d_score.squeeze()

    class CBiGANTrainer:
        """Trainer for the CBiGAN model."""

        def __init__(self, pipeline: "CBiGANDataAugPipeline", latent_dim: int = 64, lr: float = 0.0002, device: torch.device | str = "cuda") -> None:
            super().__init__()
            self.pipeline = pipeline
            self.device = torch.device(device)
            self.latent_dim = latent_dim

            self.G = pipeline.Generator(latent_dim).to(self.device)
            self.E = pipeline.Encoder(latent_dim).to(self.device)
            self.D = pipeline.Discriminator(latent_dim).to(self.device)

            self.opt_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
            self.opt_E = torch.optim.Adam(self.E.parameters(), lr=lr, betas=(0.5, 0.999))
            self.opt_D = torch.optim.Adam(self.D.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))

            self.criterion = nn.BCEWithLogitsLoss()
            self.metrics = pipeline.AnomalyMetrics(self.device)
            self.history: dict[str, list[float]] = {"d_loss": [], "ge_loss": [], "recon_error": []}

        def train(self, train_loader: DataLoader, num_epochs: int = 50, print_every: int = 5) -> None:
            for epoch in range(num_epochs):
                epoch_d_loss = 0.0
                epoch_ge_loss = 0.0
                epoch_recon_error = 0.0

                for batch_idx, real_images in enumerate(train_loader):
                    real_images = real_images.to(self.device)
                    batch_size = real_images.size(0)

                    if batch_idx % 2 == 0:
                        self.opt_D.zero_grad()

                        real_z = self.E(real_images)
                        d_real = self.D(real_images, real_z.detach())

                        fake_z = torch.randn(batch_size, self.latent_dim, device=self.device)
                        fake_images = self.G(fake_z)
                        d_fake = self.D(fake_images.detach(), fake_z)

                        real_labels = torch.ones_like(d_real) * 0.9
                        fake_labels = torch.zeros_like(d_fake) * 0.1

                        d_loss = (self.criterion(d_real, real_labels) + self.criterion(d_fake, fake_labels)) / 2

                        d_loss.backward()
                        self.opt_D.step()
                        epoch_d_loss += d_loss.item()

                    self.opt_G.zero_grad()
                    self.opt_E.zero_grad()

                    real_z_ge = self.E(real_images)
                    fake_z_ge = torch.randn(batch_size, self.latent_dim, device=self.device)
                    fake_images_ge = self.G(fake_z_ge)

                    d_fake_ge = self.D(fake_images_ge, fake_z_ge)
                    d_real_ge = self.D(real_images, real_z_ge)

                    adv_loss = (self.criterion(d_fake_ge, torch.ones_like(d_fake_ge)) + self.criterion(d_real_ge, torch.zeros_like(d_real_ge))) / 2

                    reconstructed = self.G(real_z_ge)
                    recon_loss = F.mse_loss(real_images, reconstructed) * 10

                    ge_loss = adv_loss + recon_loss

                    ge_loss.backward()
                    self.opt_G.step()
                    self.opt_E.step()

                    epoch_ge_loss += ge_loss.item()
                    epoch_recon_error += recon_loss.item()

                self.history["d_loss"].append(epoch_d_loss / len(train_loader))
                self.history["ge_loss"].append(epoch_ge_loss / len(train_loader))
                self.history["recon_error"].append(epoch_recon_error / len(train_loader))

                if (epoch + 1) % print_every == 0:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}] | "
                        f"D Loss: {self.history['d_loss'][-1]:.4f} | "
                        f"GE Loss: {self.history['ge_loss'][-1]:.4f} | "
                        f"Recon: {self.history['recon_error'][-1]:.4f}"
                    )

        def detect_anomalies(self, test_loader: DataLoader) -> pd.DataFrame:
            self.G.eval()
            self.E.eval()
            self.D.eval()

            results: list[dict[str, float]] = []

            with torch.no_grad():
                for images in test_loader:
                    images = images.to(self.device)

                    z = self.E(images)
                    reconstructed = self.G(z)

                    recon_error = self.metrics.reconstruction_error(images, reconstructed)
                    ssim_scores = self.metrics.ssim_score(images, reconstructed)
                    disc_error = self.metrics.discriminator_error(self.D, images, z)

                    for i in range(images.size(0)):
                        results.append(
                            {
                                "reconstruction_error": recon_error[i].item(),
                                "ssim_score": ssim_scores[i].item(),
                                "discriminator_error": disc_error[i].item(),
                                "combined_score": (
                                    recon_error[i].item()
                                    + (1 - ssim_scores[i].item())
                                    + disc_error[i].item()
                                )
                                / 3,
                            }
                        )

            self.G.train()
            self.E.train()
            self.D.train()

            return pd.DataFrame(results)

        def plot_training_history(self) -> None:
            epochs = range(1, len(self.history["d_loss"]) + 1)

            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(epochs, self.history["d_loss"], label="D Loss")
            plt.plot(epochs, self.history["ge_loss"], label="GE Loss")
            plt.title("Training Losses")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()

            plt.subplot(1, 3, 2)
            plt.plot(epochs, self.history["recon_error"])
            plt.title("Reconstruction Error")
            plt.xlabel("Epochs")
            plt.ylabel("MSE")

            plt.subplot(1, 3, 3)
            self.show_reconstructions()

            plt.tight_layout()
            plt.show()

        def show_reconstructions(self, num_examples: int = 4) -> None:
            self.G.eval()
            self.E.eval()

            z_sample = torch.randn(num_examples, self.latent_dim, device=self.device)

            with torch.no_grad():
                generated = self.G(z_sample)

            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            axes = axes.flatten()

            for i in range(num_examples):
                img = generated[i].squeeze().cpu().numpy()
                img = (img + 1) / 2
                axes[i].imshow(img, cmap="gray")
                axes[i].axis("off")
                axes[i].set_title(f"Generated {i + 1}")

            plt.tight_layout()
            plt.show()

        def visualize_anomaly_detection(self, test_loader: DataLoader, results_df: pd.DataFrame, num_examples: Optional[int] = None) -> None:
            self.pipeline.create_directory(self.pipeline.output_dir_images)

            results_df_sorted = results_df.sort_values(by="combined_score", ascending=False)

            if num_examples is None:
                examples_to_show = results_df_sorted
            else:
                examples_to_show = results_df_sorted.head(num_examples)

            with torch.no_grad():
                for idx, (_, row) in enumerate(examples_to_show.iterrows()):
                    batch_idx = 0
                    original = None

                    for i, images_batch in enumerate(test_loader):
                        if batch_idx * test_loader.batch_size + len(images_batch) > row.name:
                            img_in_batch_idx = row.name - batch_idx * test_loader.batch_size
                            original = images_batch[img_in_batch_idx : img_in_batch_idx + 1].to(self.device)
                            break
                        batch_idx += 1

                    if original is not None:
                        z = self.E(original)
                        reconstructed = self.G(z)

                        orig_np_ssim = (original.squeeze().cpu().numpy() + 1) / 2
                        recon_np_ssim = (reconstructed.squeeze().cpu().numpy() + 1) / 2
                        _, ssim_map = ssim(orig_np_ssim, recon_np_ssim, data_range=1.0, full=True)
                        ssim_error_map = 1 - ssim_map

                        orig_np = original.squeeze().cpu().numpy()
                        recon_np = reconstructed.squeeze().cpu().numpy()

                        orig_viz = (orig_np + 1) / 2
                        recon_viz = (recon_np + 1) / 2

                        filename = (
                            f"reconstructed_anomaly_score_{row['combined_score']:.4f}_"
                            f"ssim_{row['ssim_score']:.4f}_{row.name}.png"
                        )
                        self.pipeline.save_image(recon_viz, self.pipeline.output_dir_images, filename)

                        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                        axes[0].imshow(orig_viz, cmap="gray")
                        axes[0].set_title(f"Original {row.name}")
                        axes[0].axis("off")

                        axes[1].imshow(recon_viz, cmap="gray")
                        axes[1].set_title(
                            f"Reconstructed\nScore: {row['combined_score']:.3f}\nSSIM: {row['ssim_score']:.3f}"
                        )
                        axes[1].axis("off")

                        axes[2].imshow(ssim_error_map, cmap="hot")
                        axes[2].set_title("SSIM Error Map")
                        axes[2].axis("off")

                        plt.tight_layout()
                        plt.show()
                    else:
                        print(f"Warning: Could not locate image with index {row.name} in the test loader.")

    def __init__(
        self,
        dataset_path: str,
        latent_dim: int = 64,
        batch_size: int = 32,
        num_epochs: int = 1000,
        output_dir_images: str | Path = "reconstructions",
        output_dir_grids: str | Path = "image_samples",
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.dataset_path = dataset_path
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.output_dir_images = Path(output_dir_images)
        self.output_dir_grids = Path(output_dir_grids)

        self.create_directory(self.output_dir_images)
        self.create_directory(self.output_dir_grids)

    @staticmethod
    def create_directory(directory: str | Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_image(image: np.ndarray, path: str | Path, filename: str) -> None:
        file_path = Path(path) / filename
        print(f"Saving image at: {file_path}")
        plt.imsave(file_path, image, cmap="gray")
        print(f"Saved image: {filename}")

    def create_transforms(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.RandomApply([self.RandomRotation(max_angle=180)], p=0.7),
                transforms.RandomApply([self.RandomFlip()], p=0.6),
                transforms.RandomApply([self.RandomBrightness(brightness_range=(0.8, 1.2))], p=0.5),
                self.Rescale((128, 128)),
                self.Normalize(),
                self.ToTensor(device=self.device),
            ]
        )

    def create_dataloaders(
        self,
        dataset: Dataset,
        train_ratio: float = 0.8,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
    ) -> tuple[DataLoader, DataLoader]:
        if batch_size is None:
            batch_size = self.batch_size

        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = total_size - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available() and num_workers > 0,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available() and num_workers > 0,
        )

        print("Dataset split:")
        print(f"  Training: {len(train_dataset)} images ({len(train_dataset) / total_size * 100:.1f}%)")
        print(f"  Validation: {len(val_dataset)} images ({len(val_dataset) / total_size * 100:.1f}%)")
        print(f"  Batch size: {batch_size}")

        return train_dataloader, val_dataloader

    def run_anomaly_detection(self) -> tuple["CBiGANDataAugPipeline.CBiGANTrainer", pd.DataFrame]:
        print("Starting CBiGAN anomaly detection pipeline.")

        transform_pipeline = self.create_transforms()
        print("\nLoading dataset...")
        dataset = self.FitsDataset(self.dataset_path, transform=transform_pipeline)

        train_loader, val_loader = self.create_dataloaders(dataset, batch_size=self.batch_size)

        print("\nInitializing models...")
        trainer = self.CBiGANTrainer(self, latent_dim=self.latent_dim, device=self.device)

        print("\nTraining model...")
        trainer.train(train_loader, num_epochs=self.num_epochs, print_every=5)

        print("\nDetecting anomalies...")
        results_df = trainer.detect_anomalies(val_loader)

        print("\nResults:")
        print(results_df.describe())

        trainer.plot_training_history()
        trainer.visualize_anomaly_detection(val_loader, results_df)

        print("\nSaving model...")
        torch.save(
            {
                "generator": trainer.G.state_dict(),
                "encoder": trainer.E.state_dict(),
                "discriminator": trainer.D.state_dict(),
            },
            "cbigan_anomaly_model.pth",
        )

        print("Training completed.")

        return trainer, results_df

    def analyze_anomaly_results(self, results_df: pd.DataFrame, threshold_percentile: float = 95) -> pd.DataFrame:
        threshold = results_df["combined_score"].quantile(threshold_percentile / 100)
        anomalies = results_df[results_df["combined_score"] >= threshold]

        print("\n=== ANOMALY ANALYSIS ===")
        print(f"Anomaly threshold (percentile {threshold_percentile}): {threshold:.4f}")
        print(f"Total number of images: {len(results_df)}")
        print(f"Detected anomalies: {len(anomalies)} ({len(anomalies) / len(results_df) * 100:.1f}%)")

        print("\n=== METRIC STATISTICS ===")
        metrics_stats = results_df[
            [
                "reconstruction_error",
                "ssim_score",
                "discriminator_error",
                "combined_score",
            ]
        ].describe()
        print(metrics_stats)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].hist(results_df["reconstruction_error"], bins=30, alpha=0.7, color="blue")
        axes[0, 0].set_title("Reconstruction Error")
        axes[0, 0].set_xlabel("MSE")

        axes[0, 1].hist(results_df["ssim_score"], bins=30, alpha=0.7, color="green")
        axes[0, 1].set_title("SSIM Scores")
        axes[0, 1].set_xlabel("SSIM")

        axes[1, 0].hist(results_df["discriminator_error"], bins=30, alpha=0.7, color="orange")
        axes[1, 0].set_title("Discriminator Error")
        axes[1, 0].set_xlabel("Error")

        axes[1, 1].hist(results_df["combined_score"], bins=30, alpha=0.7, color="red")
        axes[1, 1].axvline(threshold, color="black", linestyle="--", label=f"Threshold ({threshold_percentile}%)")
        axes[1, 1].set_title("Combined Score")
        axes[1, 1].set_xlabel("Score")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

        return anomalies


if __name__ == "__main__":
    pipeline = CBiGANDataAugPipeline(dataset_path="./dataset01/")
    trainer, results = pipeline.run_anomaly_detection()
    anomalies = pipeline.analyze_anomaly_results(results)
