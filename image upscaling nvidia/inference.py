import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import time
import os
from model import UltraFastUpscaler

# --- CONFIG ---
MODEL_PATH = r"fast_upscaler_ep19.pth" # Update this to your new trained model!
TEST_FOLDER = r"D:\vimeo_super_resolution_test\low_resolution\00007\0236"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_RESOLUTION = (1920, 1080) 

# --- LOAD MODEL ---
model = UltraFastUpscaler().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- LOAD 7 FRAMES ---
to_tensor = T.ToTensor()
frames = []
for i in range(1, 8):
    img_path = os.path.join(TEST_FOLDER, f"im{i}.png")
    img = Image.open(img_path).convert('RGB')
    frames.append(to_tensor(img))

img_tensor = torch.cat(frames, dim=0).unsqueeze(0).to(DEVICE)
print(f"Input Resolution: {img.size[0]}x{img.size[1]} (Using 7 temporal frames)")

# --- GPU WARMUP (Do not measure this) ---
print("Warming up GPU...")
with torch.inference_mode():
    for _ in range(5):
        _ = model(img_tensor)
torch.cuda.synchronize() # Wait for GPU to finish warm up

# --- THE REAL TEST ---
with torch.inference_mode():
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    # AI 4x upscale using all 7 frames
    ai_4x_output = model(img_tensor)
    
    # Mathematical snap to target
    final_output = F.interpolate(
        ai_4x_output, 
        size=(TARGET_RESOLUTION[1], TARGET_RESOLUTION[0]),
        mode='bilinear', 
        align_corners=False
    )
    final_output = final_output.clamp(0, 1)
    
    torch.cuda.synchronize() # Wait for GPU to finish the real math
    end_time = time.perf_counter()

# --- RESULTS ---
execution_time_ms = (end_time - start_time) * 1000
print(f"⚡ True Pipeline Execution Time: {execution_time_ms:.2f} ms")

output_img = T.ToPILImage()(final_output.squeeze(0).cpu())
output_img.save("temporal_upscale_result.png")
print("✅ Saved as temporal_upscale_result.png")