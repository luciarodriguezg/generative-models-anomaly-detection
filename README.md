# Astronomical Image Reconstruction for Anomaly Detection

This repository contains the code developed for the Master’s Thesis in Artificial Intelligence at UNIR (Universidad Internacional de La Rioja), titled **"Reconstrucción de imágenes astronómicas con CBiGAN y DDPM orientadas a la detección de anomalías"**.

![CBiGAN best results](results\cbigan_sinaug_best10.png)

The project studies the use of **generative models**—specifically **CBiGAN (Consistency Bidirectional Generative Adversarial Networks)** and **DDPM (Denoising Diffusion Probabilistic Models)**—for **unsupervised anomaly detection in galaxy images**. Using datasets such as the COSMOS Galaxy Set, the work focuses on how well these models can reconstruct astronomical images and highlight deviations from normal behaviour that may correspond to anomalous objects.

Throughout the thesis, different **data augmentation strategies** are explored to enrich the training data without requiring explicit labels. The models are evaluated with metrics such as **Mean Squared Error (MSE)**, **PSNR**, **SSIM**, and a **global anomaly score**, allowing a quantitative comparison of reconstruction quality and anomaly detection performance. Although both CBiGAN and DDPM achieve competitive results, the study finds that a CBiGAN configuration without data augmentation provides particularly stable and accurate reconstructions.

Overall, the work proposes a **robust and scalable approach** to detect anomalies in astronomical images using generative models, aiming to support the analysis of large astronomical surveys without manual supervision.

## Repository Structure

- `cbigan_model.py`: CBiGAN training and evaluation pipeline, including hyperparameter tuning utilities and metrics for anomaly detection.
- `cbigan_DataAug_model.py`: CBiGAN pipeline variant focused on data augmentation and reconstruction-based anomaly scoring.



## How to Use

1. Create and activate a Python environment (Python 3.10+ recommended).
2. Install the required dependencies, for example:

```bash
pip install torch torchvision numpy pandas matplotlib astropy scikit-image tqdm
```

3. Prepare a directory with astronomical images in **FITS** format.
4. Update the dataset path in the scripts (for example in `cbigan_model.py` or `cbigan_DataAug_model.py`) to point to your local dataset.
5. Run the desired script or notebook to train CBiGAN and compute anomaly scores.

## Authors

- **Lucía Isabel Rodríguez González**
- **Ibai Carracedo Sánchez**
- **Sandra Sánchez Prevost**

Master’s Thesis in Artificial Intelligence – UNIR.
