import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
from ultralytics import YOLO
import os

st.set_page_config(page_title="PCB Defect Detection", layout="wide")
st.title("📦 PCB Defect Detection using YOLOv8")

# Use YOLO model in root folder
YOLO_PATH = "yolov8s (1).pt"

# Validate model
if not os.path.exists(YOLO_PATH):
    st.error("❌ YOLO model not found. Please upload 'yolov8s (1).pt' to the root of your repo.")
    st.stop()

# Load model
yolo_model = YOLO(YOLO_PATH)

# Upload UI
st.header("Upload a PCB Image")
uploaded_file = st.file_uploader("Upload a .jpg/.jpeg/.png file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Try font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except:
        font = ImageFont.load_default()

    # Run YOLO detection
    results = yolo_model.predict(source=image, device=0, conf=0.25)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        st.warning("❌ No defects detected.")
    else:
        st.info(f"✅ Detected {len(boxes)} defects:")
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            text = f"Defect: {label}"

            # Label background
            text_size = draw.textbbox((x1, y1), text, font=font)
            text_width = text_size[2] - text_size[0]
            text_height = text_size[3] - text_size[1]
            draw.rectangle(
                [(x1, y1 - text_height - 10), (x1 + text_width + 10, y1)],
                fill="white"
            )
            draw.text((x1 + 5, y1 - text_height - 5), text, fill="black", font=font)
            draw.rectangle([(x1, y1), (x2, y2)], outline="white", width=3)

        st.image(image, caption="📍 Defects Detected", use_column_width=True)
