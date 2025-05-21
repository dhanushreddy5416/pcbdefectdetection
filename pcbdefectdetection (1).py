import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import os
import gdown

st.set_page_config(page_title="PCB Defect Detection", layout="wide")
st.title("🔍 PCB Defect Detection using YOLOv8 + DenseNet")

# Class names
class_names = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

# Paths
YOLO_PATH = "yolov8s (1).pt"
DENSENET_PATH = "densenet_best.pth"

# Download DenseNet model from Google Drive if missing
if not os.path.exists(DENSENET_PATH):
    st.info("⬇️ Downloading DenseNet model from Google Drive...")
    gdown.download(id="1wHi60pPosWaHhuwB-PNsO85ib5vsb568", output=DENSENET_PATH, quiet=False)

# Load YOLO model
if not os.path.exists(YOLO_PATH):
    st.error("❌ YOLO model not found. Please upload 'yolov8s (1).pt' to the root of your repo.")
    st.stop()

yolo_model = YOLO(YOLO_PATH)

# Load DenseNet model
densenet = models.densenet121(pretrained=False)
densenet.classifier = nn.Linear(densenet.classifier.in_features, len(class_names))
densenet.load_state_dict(torch.load(DENSENET_PATH, map_location="cpu"))
densenet.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Upload UI
st.header("Upload a PCB Image")
uploaded_img = st.file_uploader("Upload a .jpg/.jpeg/.png file", type=["jpg", "jpeg", "png"])

if uploaded_img:
    image = Image.open(uploaded_img).convert("RGB")
    draw = ImageDraw.Draw(image)

    results = yolo_model.predict(source=image, conf=0.25)
    boxes = results[0].boxes

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    if boxes and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = image.crop((x1, y1, x2, y2)).resize((224, 224))
            input_tensor = transform(cropped).unsqueeze(0)

            with torch.no_grad():
                out = densenet(input_tensor)
                label = class_names[torch.argmax(out).item()]

            draw.rectangle([(x1, y1), (x2, y2)], outline="white", width=3)
            draw.text((x1, y1 - 20), label, fill="white", font=font)

        st.image(image, caption="📍 Defects Detected", use_column_width=False)
    else:
        st.warning("⚠️ No defects detected.")
