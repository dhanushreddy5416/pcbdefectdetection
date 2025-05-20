import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

st.set_page_config(page_title="PCB Defect Detection", layout="wide")
st.title("🔍 PCB Defect Detection (No Model Download Required)")

# Load YOLOv8 model from Ultralytics pretrained model (you can fine-tune later)
yolo_model = YOLO("yolov8s.pt")  # small model, pretrained on COCO

# Simulated class names for testing
class_names = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

# Load DenseNet121 from torchvision and replace classifier
densenet_model = models.densenet121(pretrained=True)
densenet_model.classifier = nn.Linear(densenet_model.classifier.in_features, len(class_names))
densenet_model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Upload image
st.header("Upload a PCB Image")
uploaded_img = st.file_uploader("Upload a .jpg/.png image", type=["jpg", "jpeg", "png"])

if uploaded_img:
    image = Image.open(uploaded_img).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Detect with YOLO
    results = yolo_model.predict(source=image, conf=0.25)
    boxes = results[0].boxes

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    if boxes:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image.crop((x1, y1, x2, y2)).resize((224, 224))
            input_tensor = transform(crop).unsqueeze(0)

            with torch.no_grad():
                out = densenet_model(input_tensor)
                label = class_names[torch.argmax(out).item()]

            draw.rectangle([(x1, y1), (x2, y2)], outline="white", width=3)
            draw.text((x1, y1 - 20), label, fill="white", font=font)

        st.image(image, caption="🔍 Defects Detected", use_column_width=False)
    else:
        st.warning("❌ No defects detected.")
