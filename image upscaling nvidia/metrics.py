import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import time
import math
import pandas as pd
import matplotlib.pyplot as plt

from dataset import VimeoDataset
from model import ESPCN

# ==========================
# OPTIONAL LPIPS
# ==========================
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("LPIPS not installed. Run: pip install lpips")

# ==========================
# PATHS
# ==========================
DATA_ROOT = r"D:\vimeo_super_resolution_test"
MODEL_PATH = r"D:\RealTimeVideoUpScaleAndFrameGenaration\image upscaling nvidia\espcn_vimeo_final.pth"

SAVE_OUTPUT_DIR = "metric_outputs"
PLOT_DIR = os.path.join(SAVE_OUTPUT_DIR, "plots")
CSV_PATH = os.path.join(SAVE_OUTPUT_DIR, "per_image_metrics.csv")

os.makedirs(SAVE_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ==========================
# SETTINGS
# ==========================
BATCH_SIZE = 1
SCALE_FACTOR = 4
NUM_WORKERS = 0   # safer on Windows


def calculate_psnr(output, target):
    mse = torch.mean((output - target) ** 2)
    if mse.item() == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))


def calculate_ssim_simple(output, target):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = output.mean()
    mu_y = target.mean()

    sigma_x = output.var()
    sigma_y = target.var()
    sigma_xy = ((output - mu_x) * (target - mu_y)).mean()

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    )

    return ssim.item()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def save_distribution_plot(values, title, xlabel, ylabel, filename, bins=30):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()


def save_line_plot(x, y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker="o", linewidth=1.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()


def save_bar_plot(names, values, title, xlabel, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading dataset...")
    dataset = VimeoDataset(root_dir=DATA_ROOT)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(f"Found {len(dataset)} image pairs.")

    print("Loading model...")
    model = ESPCN(scale_factor=SCALE_FACTOR).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("Model loaded successfully.")

    # ==========================
    # LPIPS MODEL
    # ==========================
    if LPIPS_AVAILABLE:
        lpips_model = lpips.LPIPS(net="alex").to(device)
        lpips_model.eval()
        print("LPIPS model loaded successfully.")
    else:
        lpips_model = None

    total_psnr = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_lpips = 0.0
    total_time = 0.0

    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()

    num_images = len(dataloader)

    # ==========================
    # STORE VALUES FOR PLOTS
    # ==========================
    image_indices = []
    psnr_values = []
    ssim_values = []
    mse_values = []
    mae_values = []
    rmse_values = []
    lpips_values = []
    latency_ms_values = []
    fps_values = []

    with torch.no_grad():
        for idx, (lr_img, hr_img) in enumerate(dataloader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.time()
            output = model(lr_img)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.time()

            inference_time = end - start
            inference_time_ms = inference_time * 1000
            fps_single = 1000 / inference_time_ms if inference_time_ms > 0 else 0

            total_time += inference_time

            output = torch.clamp(output, 0.0, 1.0)
            hr_img = torch.clamp(hr_img, 0.0, 1.0)

            mse = mse_loss(output, hr_img).item()
            mae = mae_loss(output, hr_img).item()
            rmse = math.sqrt(mse)
            psnr = calculate_psnr(output, hr_img)
            ssim = calculate_ssim_simple(output, hr_img)

            # LPIPS expects images in range [-1, 1]
            if lpips_model is not None:
                output_lpips = output * 2 - 1
                hr_lpips = hr_img * 2 - 1
                lpips_score = lpips_model(output_lpips, hr_lpips).item()
            else:
                lpips_score = float("nan")

            total_mse += mse
            total_mae += mae
            total_rmse += rmse
            total_psnr += psnr
            total_ssim += ssim

            if not math.isnan(lpips_score):
                total_lpips += lpips_score

            image_indices.append(idx + 1)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            mse_values.append(mse)
            mae_values.append(mae)
            rmse_values.append(rmse)
            lpips_values.append(lpips_score)
            latency_ms_values.append(inference_time_ms)
            fps_values.append(fps_single)

            if idx < 10:
                save_image(lr_img.cpu(), os.path.join(SAVE_OUTPUT_DIR, f"{idx+1}_LR.png"))
                save_image(output.cpu(), os.path.join(SAVE_OUTPUT_DIR, f"{idx+1}_SR_Output.png"))
                save_image(hr_img.cpu(), os.path.join(SAVE_OUTPUT_DIR, f"{idx+1}_HR_Target.png"))

            if idx % 50 == 0:
                print(
                    f"[{idx}/{num_images}] "
                    f"PSNR: {psnr:.2f} dB | "
                    f"SSIM: {ssim:.4f} | "
                    f"LPIPS: {lpips_score:.4f} | "
                    f"Latency: {inference_time_ms:.2f} ms | "
                    f"FPS: {fps_single:.2f}"
                )

    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    avg_mse = total_mse / num_images
    avg_mae = total_mae / num_images
    avg_rmse = total_rmse / num_images
    avg_time_ms = (total_time / num_images) * 1000
    avg_fps = 1000 / avg_time_ms

    if LPIPS_AVAILABLE:
        avg_lpips = total_lpips / num_images
    else:
        avg_lpips = float("nan")

    params = count_parameters(model)
    model_size = get_model_size_mb(MODEL_PATH)

    # ==========================
    # SAVE CSV
    # ==========================
    df = pd.DataFrame({
        "Image_Index": image_indices,
        "PSNR_dB": psnr_values,
        "SSIM": ssim_values,
        "LPIPS": lpips_values,
        "MSE": mse_values,
        "MAE": mae_values,
        "RMSE": rmse_values,
        "Latency_ms": latency_ms_values,
        "FPS": fps_values
    })

    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved per-image metrics CSV: {CSV_PATH}")

    # ==========================
    # PLOTS
    # ==========================

    # 1. PSNR distribution
    save_distribution_plot(
        psnr_values,
        title="PSNR Distribution",
        xlabel="PSNR (dB)",
        ylabel="Number of Images",
        filename="psnr_distribution.png"
    )

    # 2. SSIM distribution
    save_distribution_plot(
        ssim_values,
        title="SSIM Distribution",
        xlabel="SSIM",
        ylabel="Number of Images",
        filename="ssim_distribution.png"
    )

    # 3. LPIPS distribution
    clean_lpips_values = [v for v in lpips_values if not math.isnan(v)]

    if len(clean_lpips_values) > 0:
        save_distribution_plot(
            clean_lpips_values,
            title="LPIPS Distribution",
            xlabel="LPIPS",
            ylabel="Number of Images",
            filename="lpips_distribution.png"
        )
    else:
        print("Skipping LPIPS distribution plot because LPIPS is not available.")

    # 4. Latency graph
    save_line_plot(
        image_indices,
        latency_ms_values,
        title="Per-Image Inference Latency",
        xlabel="Image Index",
        ylabel="Latency (ms)",
        filename="latency_graph.png"
    )

    # 5. FPS comparison
    # Currently only ESPCN is tested in this code.
    # You can manually add other models later.
    model_names = ["ESPCN"]
    fps_comparison = [avg_fps]

    save_bar_plot(
        model_names,
        fps_comparison,
        title="FPS Comparison",
        xlabel="Model",
        ylabel="FPS",
        filename="fps_comparison.png"
    )

    # 6. Model size comparison
    # Currently only ESPCN model size is available.
    # You can manually add other models later.
    model_size_comparison = [model_size]

    save_bar_plot(
        model_names,
        model_size_comparison,
        title="Model Size Comparison",
        xlabel="Model",
        ylabel="Model Size (MB)",
        filename="model_size_comparison.png"
    )

    # 7. Optional combined quality comparison
    save_bar_plot(
        ["PSNR", "SSIM", "LPIPS"],
        [avg_psnr, avg_ssim, avg_lpips if not math.isnan(avg_lpips) else 0],
        title="Average Quality Metrics",
        xlabel="Metric",
        ylabel="Value",
        filename="average_quality_metrics.png"
    )

    print("\n" + "=" * 60)
    print("                 ESPCN METRICS REPORT")
    print("=" * 60)
    print(f"Model path:              {MODEL_PATH}")
    print(f"Dataset path:            {DATA_ROOT}")
    print(f"Total test images:       {num_images}")
    print(f"Device:                  {device}")
    print("-" * 60)
    print(f"Average PSNR:            {avg_psnr:.2f} dB")
    print(f"Average SSIM:            {avg_ssim:.4f}")

    if LPIPS_AVAILABLE:
        print(f"Average LPIPS:           {avg_lpips:.4f}")
    else:
        print("Average LPIPS:           Not calculated")

    print(f"Average MSE:             {avg_mse:.6f}")
    print(f"Average MAE:             {avg_mae:.6f}")
    print(f"Average RMSE:            {avg_rmse:.6f}")
    print("-" * 60)
    print(f"Inference time/image:    {avg_time_ms:.2f} ms")
    print(f"Estimated FPS:           {avg_fps:.2f}")
    print(f"Trainable parameters:    {params:,}")
    print(f"Model size:              {model_size:.2f} MB")
    print("-" * 60)
    print(f"Saved sample outputs in: {SAVE_OUTPUT_DIR}")
    print(f"Saved plots in:          {PLOT_DIR}")
    print(f"Saved CSV file:          {CSV_PATH}")
    print("=" * 60)

    print("\nGenerated Plots:")
    print("1. psnr_distribution.png")
    print("2. ssim_distribution.png")
    print("3. lpips_distribution.png")
    print("4. latency_graph.png")
    print("5. fps_comparison.png")
    print("6. model_size_comparison.png")
    print("7. average_quality_metrics.png")

    print("\nInterpretation:")
    if avg_psnr > 35:
        print("PSNR: Excellent")
    elif avg_psnr > 30:
        print("PSNR: Good")
    elif avg_psnr > 25:
        print("PSNR: Acceptable")
    else:
        print("PSNR: Needs improvement")

    if avg_ssim > 0.95:
        print("SSIM: Excellent")
    elif avg_ssim > 0.85:
        print("SSIM: Good")
    elif avg_ssim > 0.70:
        print("SSIM: Acceptable")
    else:
        print("SSIM: Needs improvement")

    if LPIPS_AVAILABLE:
        if avg_lpips < 0.05:
            print("LPIPS: Excellent perceptual similarity")
        elif avg_lpips < 0.15:
            print("LPIPS: Good perceptual similarity")
        elif avg_lpips < 0.30:
            print("LPIPS: Acceptable perceptual similarity")
        else:
            print("LPIPS: Needs improvement")

    if avg_time_ms < 16:
        print("Speed: 60 FPS capable")
    elif avg_time_ms < 33:
        print("Speed: 30 FPS capable")
    else:
        print("Speed: Not real-time yet")


if __name__ == "__main__":
    main()