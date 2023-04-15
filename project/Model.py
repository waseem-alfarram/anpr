from ultralytics import YOLO
import torch

# Use the weights previously obtained from training.
model = YOLO('./weights/best.pt')
# If a supported GPU is avaliable, use it instead of cpu.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def predict(source):
    results = model(source=source, stream=True)
    output = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            output.append(box.xyxy[0].numpy().astype(int))
    return output