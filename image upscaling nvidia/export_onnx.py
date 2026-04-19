import torch
import torch.onnx
from model import ESPCN  # Importing your residual model architecture

# ==============================================================================
# ⚠️ PATHS: UPDATE THESE IF NECESSARY ⚠️
# ==============================================================================
LOAD_MODEL_PATH = "espcn_vimeo_final.pth"
ONNX_OUTPUT_PATH = "espcn_4x_dynamic.onnx"
# ==============================================================================

def export_to_onnx():
    # 1. Load the PyTorch Model
    print("Loading PyTorch model...")
    device = torch.device("cpu") # Exporting on CPU is standard and prevents device mismatch errors
    
    model = ESPCN(scale_factor=4).to(device)
    model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location=device, weights_only=True))
    
    # CRITICAL: Put the model in evaluation mode before exporting!
    model.eval() 

    # 2. Create a Dummy Input
    # The ONNX exporter needs a sample tensor to trace the math operations.
    # The exact size here (64x64) doesn't matter because we will make it dynamic later.
    dummy_input = torch.randn(1, 3, 64, 64, device=device)

    # 3. Define Dynamic Axes
    # This tells ONNX that the Batch Size, Height, and Width can change during inference.
    dynamic_axes = {
        'im_lr': {
            0: 'batch_size', 
            2: 'height', 
            3: 'width'
        },
        'pred_hr': {
            0: 'batch_size', 
            2: 'height_out', 
            3: 'width_out'
        }
    }

    # 4. Export the Model
    print(f"Exporting to {ONNX_OUTPUT_PATH}...")
    torch.onnx.export(
        model,                         # The loaded PyTorch model
        dummy_input,                   # The dummy tensor to trace
        ONNX_OUTPUT_PATH,              # Where to save the file
        export_params=True,            # Store the trained weights inside the ONNX file
        opset_version=14,              # Opset 14 is highly stable for modern PyTorch operations
        do_constant_folding=True,      # Optimizes the math for faster inference
        input_names=['im_lr'],         # Name of the input tensor (matches your teammate's specs)
        output_names=['pred_hr'],      # Name of the output tensor
        dynamic_axes=dynamic_axes      # Apply the dynamic sizes we defined above
    )
    
    print("✅ ONNX export complete! Your model is ready for the pipeline.")

if __name__ == "__main__":
    export_to_onnx()