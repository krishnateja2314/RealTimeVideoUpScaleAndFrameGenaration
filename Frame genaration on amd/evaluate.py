import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch.utils.data import DataLoader
from dataset import VimeoDataset
from model import InterpolationModel

# Corrected Imports for 2026 torchmetrics versions
from torchmetrics.image import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure, 
    MultiScaleStructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity
)
from torchmetrics.regression import MeanSquaredError

def save_visual_comparison(im1, pred, im3, gt, idx, save_path="results/visuals"):
    os.makedirs(save_path, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    titles = ["Frame 1 (t=0)", "Generated (t=0.5)", "Ground Truth", "Frame 3 (t=1)"]
    imgs = [im1, pred, gt, im3]
    
    for ax, img, title in zip(axes, imgs, titles):
        curr_img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        ax.imshow(np.clip(curr_img, 0, 1))
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/sample_{idx}.png")
    plt.close()

def main():
    # Use GPU (9070 XT) via ROCm
    device = torch.device('cuda')
    DATA_ROOT = "/run/media/krishnateja/Coding/Courses/ml for physics/Project/amd/data/vimeo_triplet/sequences"
    TEST_LIST = "/run/media/krishnateja/Coding/Courses/ml for physics/Project/amd/data/vimeo_triplet/tri_testlist.txt"

    dataset = VimeoDataset(DATA_ROOT, TEST_LIST)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = InterpolationModel().to(device)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    # --- 1-6. Mathematical & Perceptual Metrics ---
    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ms_ssim_m = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    mse_m = MeanSquaredError().to(device)
    rmse_m = MeanSquaredError(squared=False).to(device) # Interpolation Error (IE)

    results = []
    print(f"Benchmarking all {len(dataset)} samples on 9070 XT... (This will take a moment)")

    with torch.no_grad():
        for i, (im1, im3, im2) in enumerate(loader):
            im1, im3, im2 = im1.to(device), im3.to(device), im2.to(device)

            # --- 7-8. Performance Metrics (Latency & Throughput) ---
            torch.cuda.synchronize() 
            start_time = time.time()
            pred = model(im1, im3)
            torch.cuda.synchronize()
            latency = (time.time() - start_time) * 1000 # ms

            # Metric Calculations
            p = psnr_m(pred, im2).item()
            s = ssim_m(pred, im2).item()
            ms = ms_ssim_m(pred, im2).item()
            l = lpips_m(pred, im2).item()
            mse_val = mse_m(pred, im2).item()
            ie_val = rmse_m(pred, im2).item()

            results.append({
                "sample": i, "psnr": p, "ssim": s, "ms_ssim": ms, 
                "lpips": l, "mse": mse_val, "ie": ie_val, "latency": latency
            })

            # Print progress every 200 samples so your terminal isn't spammed
            if i % 200 == 0:
                save_visual_comparison(im1, pred, im3, im2, i)
                print(f"[{i}/{len(dataset)}] PSNR: {p:.2f} | Latency: {latency:.2f}ms")

    # --- Data Processing ---
    print("\nProcessing data and generating graphs...")
    df = pd.DataFrame(results)
    
    # Create the output directories if they don't exist
    plots_dir = os.path.join(os.getcwd(), "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    avg_psnr = df['psnr'].mean()
    psnr_std = df['psnr'].std() 
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2 

    # --- Visual Insights for Report ---
    plt.style.use('ggplot')
    sns.set_context("paper", font_scale=1.2)
    
    # Plot 1: PSNR Stability
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x='sample', y='psnr', color='#3498db', label='Sample PSNR')
    plt.axhline(y=avg_psnr, color='#e74c3c', linestyle='--', label=f'Mean: {avg_psnr:.2f}')
    plt.fill_between(df['sample'], avg_psnr - psnr_std, avg_psnr + psnr_std, color='#e74c3c', alpha=0.15)
    plt.title("Model Stability (PSNR across sequence)")
    plt.xlabel("Sample Index")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "1_stability_analysis.png"))

    # Plot 2: Perceptual Correlation
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df["psnr"], df["lpips"], c=df["latency"], cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Latency (ms)")
    plt.title("Quality vs. Perceptual Loss (PSNR vs LPIPS)")
    plt.xlabel("PSNR (Higher is better)")
    plt.ylabel("LPIPS (Lower is better)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "2_quality_correlation.png"))

    # Plot 3: Steady-State Latency Distribution (Ignoring index 0 warmup)
    plt.figure(figsize=(8, 5))
    steady_state_df = df.iloc[1:] # Drop the first 2000+ ms warmup frame
    sns.histplot(steady_state_df['latency'], bins=40, kde=True, color='#2ecc71')
    plt.axvline(x=steady_state_df['latency'].mean(), color='black', linestyle='--', label=f"Avg: {steady_state_df['latency'].mean():.2f}ms")
    plt.title("Steady-State Latency Distribution (AMD 9070 XT)")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "3_latency_distribution.png"))

    # Plot 4: Cumulative Distribution Function (CDF) of PSNR
    plt.figure(figsize=(8, 5))
    sns.ecdfplot(data=df, x="psnr", color='#9b59b6', linewidth=2)
    plt.axvline(x=avg_psnr, color='#e74c3c', linestyle=':', label=f'Mean PSNR: {avg_psnr:.2f} dB')
    plt.title("Cumulative Distribution Function (CDF) of PSNR")
    plt.xlabel("PSNR Threshold (dB)")
    plt.ylabel("Proportion of Frames")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "4_psnr_cdf.png"))

    # Plot 5: Metric Correlation Heatmap
    plt.figure(figsize=(9, 7))
    corr_cols = ['psnr', 'ssim', 'ms_ssim', 'lpips', 'ie', 'latency']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True)
    plt.title("Correlation Heatmap of Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "5_metric_correlation.png"))

    print("\n" + "="*40)
    print("COMPREHENSIVE REPORT SUMMARY")
    print(f"1. PSNR: {avg_psnr:.4f} dB")
    print(f"2. SSIM: {df['ssim'].mean():.4f}")
    print(f"3. MS-SSIM: {df['ms_ssim'].mean():.4f}")
    print(f"4. LPIPS: {df['lpips'].mean():.4f}")
    print(f"5. MSE: {df['mse'].mean():.6f}")
    print(f"6. Interpolation Error (RMSE): {df['ie'].mean():.4f}")
    print(f"7. Avg Latency (w/ Warmup): {df['latency'].mean():.2f} ms")
    print(f"   Steady-State Latency: {steady_state_df['latency'].mean():.2f} ms")
    print(f"8. Max Throughput (Steady): {1000/steady_state_df['latency'].mean():.1f} FPS")
    print(f"9. PSNR Std Dev: {psnr_std:.4f}")
    print(f"10. Peak VRAM: {peak_vram:.2f} MB")
    print("="*40)
    
    print("\n GRAPHS GENERATED SUCCESSFULLY!")
    print(f"Please check this folder for your graphs: \n--> {plots_dir}")
    print("You can drag and drop these 5 images straight into your LaTeX report.")

if __name__ == "__main__":
    main()