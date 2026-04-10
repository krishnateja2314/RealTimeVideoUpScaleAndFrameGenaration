import os
import random
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import ESPCN
torch.backends.cudnn.benchmark = True

# -----------------------------
# SETTINGS
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "espcn.pth"

HR_DIR = r"D:\DIV2K\DIV2K_train_HR"
LR_DIR = r"D:\DIV2K\X4"

# -----------------------------
# LOAD MODEL
# -----------------------------
model = ESPCN(scale_factor=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded")

# -----------------------------
# PICK RANDOM IMAGE
# -----------------------------
lr_images = os.listdir(LR_DIR)
lr_name = random.choice(lr_images)

hr_name = lr_name.replace("x4", "")

lr_path = os.path.join(LR_DIR, lr_name)
hr_path = os.path.join(HR_DIR, hr_name)

print("🖼 Testing image:", lr_name)

# -----------------------------
# LOAD IMAGE
# -----------------------------
transform = transforms.ToTensor()

lr_img = Image.open(lr_path).convert("RGB")
lr_tensor = transform(lr_img).unsqueeze(0).to(DEVICE)

# -----------------------------
# WARM-UP (IMPORTANT)
# -----------------------------
for _ in range(10):
    _ = model(lr_tensor)

# -----------------------------
# CUDA TIMING
# -----------------------------
if DEVICE == "cuda":
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    starter.record()

    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    ender.record()
    torch.cuda.synchronize()

    time_ms = starter.elapsed_time(ender)

else:
    import time
    start = time.time()

    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    end = time.time()
    time_ms = (end - start) * 1000

print(f"🔥 Inference Time: {time_ms:.3f} ms")

# -----------------------------
# SAVE OUTPUT IMAGE
# -----------------------------
sr_tensor = sr_tensor.squeeze(0).cpu().clamp(0, 1)

to_pil = transforms.ToPILImage()
sr_img = to_pil(sr_tensor)

sr_img.save("output_sr.png")

print("✅ Super-resolved image saved as output_sr.png")