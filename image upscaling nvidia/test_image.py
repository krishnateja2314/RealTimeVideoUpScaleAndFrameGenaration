import os
import random
import glob
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from model import ESPCN

# ==============================================================================
# ⚠️ PATHS: ADD YOUR SPECIFIC DIRECTORIES HERE ⚠️
# ==============================================================================
# Point this to the testing set directory containing 'low_resolution'
TEST_DATA_ROOT = r"D:\vimeo_super_resolution_test" 
LOAD_MODEL_PATH = "espcn_vimeo_deep_final.pth" # The file created by train.py
SAVE_RESULT_DIR = "./test_results"  # Folder where the upscaled image will be saved
# ==============================================================================

def test_random_image():
    # 1. Setup Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 2. Ensure save directory exists
    os.makedirs(SAVE_RESULT_DIR, exist_ok=True)

    # 3. Load the Model
    model = ESPCN(scale_factor=4).to(device)
    if not os.path.exists(LOAD_MODEL_PATH):
        print(f"Error: Model weights '{LOAD_MODEL_PATH}' not found. Run train.py first!")
        return
        
    model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location=device, weights_only=True))
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")

    # 4. Find all random low-res im4.png files in the testing set
    lr_dir = os.path.join(TEST_DATA_ROOT, 'low_resolution')
    search_pattern = os.path.join(lr_dir, '*', '*', 'im4.png')
    lr_images = glob.glob(search_pattern)
    
    if not lr_images:
        print(f"Error: No im4.png files found in {search_pattern}")
        return

    # 5. Pick a random image
    random_img_path = random.choice(lr_images)
    print(f"Selected random image: {random_img_path}")

    # 6. Process the image
    img_pil = Image.open(random_img_path).convert('RGB')
    transform_to_tensor = ToTensor()
    transform_to_pil = ToPILImage()

    # Add a batch dimension (e.g., [3, 64, 64] becomes [1, 3, 64, 64]) and move to GPU
    input_tensor = transform_to_tensor(img_pil).unsqueeze(0).to(device)

    # 7. Run inference (No gradients needed)
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 8. Remove batch dimension and convert back to image
    output_tensor = output_tensor.squeeze(0).cpu()
    output_img_pil = transform_to_pil(output_tensor)

    # 9. Save the original and upscaled images side-by-side
    save_path = os.path.join(SAVE_RESULT_DIR, "upscaled_result.png")
    output_img_pil.save(save_path)
    
    # Also save the original low-res so you can compare them
    img_pil.save(os.path.join(SAVE_RESULT_DIR, "original_low_res.png"))
    print(f"Success! Upscaled image saved to: {SAVE_RESULT_DIR}")

if __name__ == "__main__":
    test_random_image()