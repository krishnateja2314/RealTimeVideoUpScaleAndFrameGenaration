import onnxruntime as ort
import numpy as np
from PIL import Image
import math

# Load ONNX session (can use CPU or GPU execution providers)
session = ort.InferenceSession("frame_interpolator.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1)) # HWC to CHW
    arr = np.expand_dims(arr, axis=0) # Add batch dimension
    return arr

# Load and Pad
im1_np = preprocess("test/im1.png")
im3_np = preprocess("test/im3.png")

_, _, h, w = im1_np.shape

# Calculate padding to make dimensions multiples of 32
pad_h = (32 - (h % 32)) % 32
pad_w = (32 - (w % 32)) % 32

im1_pad = np.pad(im1_np, ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='reflect')
im3_pad = np.pad(im3_np, ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='reflect')

# Run ONNX inference
inputs = {'im1': im1_pad, 'im3': im3_pad}
out_pad = session.run(['pred_im2'], inputs)[0]

# Crop back to original resolution
out = out_pad[:, :, :h, :w]

# Save output
out_img = (out[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
Image.fromarray(out_img).save("output_onnx.png")
print("Saved output_onnx.png")