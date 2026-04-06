import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import sys
from model import Upscaler

# --- CONFIGURATION ---
# --- CONFIGURATION ---
# Add the 'r' before the quotes!
MODEL_PATH = r"D:\RealTimeVideoUpScaleAndFrameGenaration\upscaler_epoch10.pth" 

TEST_IMAGE = r"D:\vimeo_super_resolution_test\vimeo_super_resolution_test\input\00014\0855\im1.png"
     # Image you want to upscale
OUTPUT_PATH = r"D:\RealTimeVideoUpScaleAndFrameGenaration\output_upscaled2.png"       # Where to save the result
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- LOAD MODEL ---
print(f"Loading model from {MODEL_PATH}...")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found at {MODEL_PATH}")
    print("Make sure training is complete and the .pth file exists!")
    sys.exit()

model = Upscaler().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Model loaded successfully!")

# --- LOAD TEST IMAGE ---
print(f"Loading test image from {TEST_IMAGE}...")

if not os.path.exists(TEST_IMAGE):
    print(f"❌ Test image not found at {TEST_IMAGE}")
    print("Put any image in the same folder and update TEST_IMAGE variable!")
    sys.exit()

img = Image.open(TEST_IMAGE).convert('RGB')
original_width, original_height = img.size
print(f"✅ Image loaded! Original size: {original_width} x {original_height}")

# --- PREPARE IMAGE ---
to_tensor = T.ToTensor()
to_image = T.ToPILImage()

# Convert to tensor and clamp
img_tensor = to_tensor(img).clamp(0, 1)

# Add batch dimension and send to GPU
img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

# --- RUN UPSCALER ---
print("🚀 Running upscaler...")

with torch.no_grad():
    output = model(img_tensor)
    output = output.clamp(0, 1)

# --- PROCESS OUTPUT ---
# Remove batch dimension and move to CPU
output = output.squeeze(0).cpu()

# Convert back to PIL image
output_img = to_image(output)
new_width, new_height = output_img.size
print(f"✅ Upscaling complete!")
print(f"Input size:  {original_width} x {original_height}")
print(f"Output size: {new_width} x {new_height}")
print(f"Scale factor: {new_width // original_width}x")

# --- SAVE OUTPUT ---
output_img.save(OUTPUT_PATH)
print(f"💾 Saved upscaled image to {OUTPUT_PATH}")

# --- DISPLAY COMPARISON ---
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(img)
axes[0].set_title(f"Original\n{original_width} x {original_height}", fontsize=14)
axes[0].axis('off')

axes[1].imshow(output_img)
axes[1].set_title(f"Upscaled by AI\n{new_width} x {new_height}", fontsize=14)
axes[1].axis('off')

plt.suptitle("Super Resolution Result", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("comparison.png")
plt.show()

print("✅ Comparison saved as comparison.png")