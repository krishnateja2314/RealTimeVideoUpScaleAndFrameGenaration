import torch
import torch_directml
from PIL import Image
import torchvision.transforms as transforms

from model import InterpolationModel

device = torch_directml.device()

model = InterpolationModel().to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

to_tensor = transforms.ToTensor()

def load_img(path):
    img = Image.open(path).convert("RGB")
    return to_tensor(img).unsqueeze(0).to(device)

im1 = load_img("test/im1.png")
im3 = load_img("test/im3.png")

with torch.no_grad():
    pred = model(im1, im3)

pred = pred.squeeze(0).cpu()
transforms.ToPILImage()(pred).save("output.png")

print("Saved output.png")