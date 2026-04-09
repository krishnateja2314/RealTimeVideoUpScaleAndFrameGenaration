import os
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import FSRCNN

# -----------------------------
# SETTINGS
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = r"D:\RealTimeVideoUpScaleAndFrameGenaration\image upscaling nvidia\fsrcnn.pth"

HR_DIR = r"D:\DIV2K\DIV2K_train_HR"
LR_DIR = r"D:\DIV2K\X4"

SAVE_OUTPUT = "output_sr.png"

# -----------------------------
# LOAD MODEL
# -----------------------------
model = FSRCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded")

# -----------------------------
# PICK RANDOM IMAGE
# -----------------------------
lr_images = os.listdir(LR_DIR)
lr_name = random.choice(lr_images)

# match HR filename
hr_name = lr_name.replace("x4", "")

lr_path = os.path.join(LR_DIR, lr_name)
hr_path = os.path.join(HR_DIR, hr_name)

print("Testing image:", lr_name)

# -----------------------------
# LOAD IMAGES
# -----------------------------
transform = transforms.ToTensor()

lr_img = Image.open(lr_path).convert("RGB")
hr_img = Image.open(hr_path).convert("RGB")

lr_tensor = transform(lr_img).unsqueeze(0).to(DEVICE)

# -----------------------------
# INFERENCE
# -----------------------------
with torch.no_grad():
    sr_tensor = model(lr_tensor)

sr_tensor = sr_tensor.squeeze(0).cpu().clamp(0, 1)

to_pil = transforms.ToPILImage()
sr_img = to_pil(sr_tensor)

# save result
sr_img.save(SAVE_OUTPUT)
print("✅ Saved SR image:", SAVE_OUTPUT)

# -----------------------------
# VISUALIZE RESULTS
# -----------------------------
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Low Resolution")
plt.imshow(lr_img)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Super Resolution (Model)")
plt.imshow(sr_img)
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Ground Truth HR")
plt.imshow(hr_img)
plt.axis("off")

plt.show()