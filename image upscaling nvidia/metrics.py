import torch
import math
import time
from PIL import Image
import torchvision.transforms as T
from model import Upscaler
import os

# --- CONFIGURATION ---
MODEL_PATH = "upscaler_epoch1.pth"  # Change to latest epoch
TEST_IMAGE = "test_image.jpg"        # Your test image
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_psnr(output, target):
    mse = torch.mean((output - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(1.0 / math.sqrt(mse.item()))
    return psnr

def calculate_ssim(output, target):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = output.mean()
    mu2 = target.mean()
    sigma1 = output.var()
    sigma2 = target.var()
    sigma12 = ((output - mu1) * (target - mu2)).mean()
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
    return ssim.item()

def measure_inference_time(model, input_tensor, runs=50):
    # Warmup GPU first
    for _ in range(5):
        with torch.no_grad():
            model(input_tensor)

    # Measure over multiple runs for accuracy
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        with torch.no_grad():
            model(input_tensor)
    torch.cuda.synchronize()
    end = time.time()

    avg_ms = ((end - start) / runs) * 1000
    return avg_ms

# --- LOAD MODEL ---
print("Loading model...")
model = Upscaler().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Model loaded!")

# --- LOAD IMAGE ---
print(f"Loading test image {TEST_IMAGE}...")
img = Image.open(TEST_IMAGE).convert('RGB')
original_size = img.size

to_tensor = T.ToTensor()
to_image = T.ToPILImage()

lr_tensor = to_tensor(img).clamp(0, 1).unsqueeze(0).to(DEVICE)

# --- RUN MODEL ---
with torch.no_grad():
    output = model(lr_tensor).clamp(0, 1)

# --- CALCULATE METRICS ---
# For PSNR and SSIM we need same size images
# Resize output back to original size for comparison
output_img = to_image(output.squeeze(0).cpu())
output_resized = output_img.resize(original_size)
output_tensor = to_tensor(output_resized).unsqueeze(0)
input_tensor_cpu = lr_tensor.cpu()

psnr = calculate_psnr(output_tensor, input_tensor_cpu)
ssim = calculate_ssim(output_tensor, input_tensor_cpu)
inference_time = measure_inference_time(model, lr_tensor)

output_size = output_img.size
scale = output_size[0] // original_size[0]

# --- PRINT RESULTS ---
print("\n" + "=" * 50)
print("           📊 MODEL METRICS REPORT")
print("=" * 50)
print(f"📁 Model:          {MODEL_PATH}")
print(f"🖼️  Input size:     {original_size[0]} x {original_size[1]}")
print(f"🖼️  Output size:    {output_size[0]} x {output_size[1]}")
print(f"🔍 Scale factor:   {scale}x upscaling")
print("-" * 50)
print(f"📈 PSNR Score:     {psnr:.2f} dB", end="  ")
if psnr > 35:
    print("🟢 Excellent!")
elif psnr > 30:
    print("🟡 Good")
elif psnr > 25:
    print("🟠 Acceptable")
else:
    print("🔴 Needs more training")

print(f"📊 SSIM Score:     {ssim:.4f}", end="    ")
if ssim > 0.95:
    print("🟢 Excellent!")
elif ssim > 0.85:
    print("🟡 Good")
elif ssim > 0.70:
    print("🟠 Acceptable")
else:
    print("🔴 Needs more training")

print(f"⚡ Inference Time: {inference_time:.1f} ms per frame", end="  ")
if inference_time < 16:
    print("🟢 60fps capable!")
elif inference_time < 33:
    print("🟢 30fps capable!")
elif inference_time < 100:
    print("🟡 Getting close to real time")
else:
    print("🔴 Not real time yet")

fps = 1000 / inference_time
print(f"🎬 Estimated FPS:  {fps:.1f} fps")
print("=" * 50)

# --- WHAT TO DO NEXT ---
print("\n💡 NEXT STEPS TO IMPROVE:")
if psnr < 30:
    print("  → Train more epochs to improve PSNR")
if ssim < 0.85:
    print("  → Add perceptual loss for better SSIM")
if inference_time > 33:
    print("  → Use TensorRT to speed up inference")
print("=" * 50)