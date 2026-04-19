import torch
from model import InterpolationModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InterpolationModel().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Dummy input representing the shape (Batch, Channels, Height, Width)
dummy_im1 = torch.randn(1, 3, 256, 256).to(device)
dummy_im3 = torch.randn(1, 3, 256, 256).to(device)

print("Exporting to ONNX...")
torch.onnx.export(
    model, 
    (dummy_im1, dummy_im3), 
    "frame_interpolator.onnx",
    export_params=True,
    opset_version=16, # Opset 16 is REQUIRED for grid_sample support
    do_constant_folding=True,
    input_names=['im1', 'im3'],
    output_names=['pred_im2'],
    dynamic_axes={
        'im1': {0: 'batch_size', 2: 'height', 3: 'width'},
        'im3': {0: 'batch_size', 2: 'height', 3: 'width'},
        'pred_im2': {0: 'batch_size', 2: 'height', 3: 'width'}
    }
)
print("Export complete! frame_interpolator.onnx generated.")