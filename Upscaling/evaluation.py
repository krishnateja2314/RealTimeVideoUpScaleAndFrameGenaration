import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import glob
import random
import os
import time

from model import RealTimeUpscaler


# =============================
# METRICS
# =============================

def calculate_psnr(sr, hr):
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_ssim(img1, img2):

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()

    sigma1 = ((img1 - mu1) ** 2).mean()
    sigma2 = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    )

    return ssim


# =============================
# MAIN EVALUATION
# =============================

def evaluate_model():

    WEIGHTS = "model_epoch_50.pth"
    DATASET = r"D:\vimeo_super_resolution_test\vimeo_super_resolution_test"

    upscale_factor = 4

    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device}")

    # -----------------------------
    # Load Model
    # -----------------------------
    model = RealTimeUpscaler(upscale_factor=upscale_factor).to(device)

    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()

    # -----------------------------
    # Pick Random Image
    # -----------------------------
    target_dir = os.path.join(DATASET, "target")

    all_hr_paths = glob.glob(
        os.path.join(target_dir, "**", "*.png"),
        recursive=True
    )

    hr_path = random.choice(all_hr_paths)
    lr_path = hr_path.replace("target", "low_resolution")

    print("\nTesting image:")
    print("HR:", hr_path)

    # -----------------------------
    # Load Images
    # -----------------------------
    hr_img = Image.open(hr_path).convert("RGB")
    lr_img = Image.open(lr_path).convert("RGB")

    to_tensor = transforms.ToTensor()

    hr_tensor = to_tensor(hr_img).unsqueeze(0).to(device)
    lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)

    # -----------------------------
    # Inference + Timing
    # -----------------------------
    with torch.no_grad():

        # GPU timing must sync
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        sr_tensor = model(lr_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()

        sr_tensor = torch.clamp(sr_tensor, 0, 1)

    inference_time = end_time - start_time
    inference_ms = inference_time * 1000
    fps = 1.0 / inference_time

    # -----------------------------
    # Metrics
    # -----------------------------
    psnr_value = calculate_psnr(sr_tensor, hr_tensor).item()
    ssim_value = calculate_ssim(sr_tensor, hr_tensor).item()

    print("\n===== Evaluation Metrics =====")
    print(f"PSNR : {psnr_value:.2f} dB")
    print(f"SSIM : {ssim_value:.4f}")
    print(f"Inference Time : {inference_ms:.2f} ms")
    print(f"Approx FPS     : {fps:.2f}")

    # -----------------------------
    # Convert to Images
    # -----------------------------
    sr_img = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu())
    lr_img_display = lr_img.resize(hr_img.size, Image.NEAREST)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(lr_img_display)
    axes[0].set_title("Low Resolution")
    axes[0].axis("off")

    axes[1].imshow(sr_img)
    axes[1].set_title(
        f"SR Output\nPSNR:{psnr_value:.2f}  SSIM:{ssim_value:.3f}\n"
        f"Time:{inference_ms:.1f}ms"
    )
    axes[1].axis("off")

    axes[2].imshow(hr_img)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("evaluation_result.png", dpi=300)

    print("\n✅ Saved evaluation_result.png")
    plt.show()


# =============================
if __name__ == "__main__":
    evaluate_model()