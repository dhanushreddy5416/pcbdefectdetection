import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os

st.set_page_config(page_title="PCB Defect Detection", layout="wide")
st.title("🔍 PCB Defect Detection using YOLOv8 (Custom Path)")

# Use the exact filename as in the user's GitHub: 'yolov8s (1).pt'
YOLO_PATH = "yolov8s (1).pt"

if not os.path.exists(YOLO_PATH):
    st.error("❌ YOLO model not found. Please make sure 'yolov8s (1).pt' is in the root of your repo.")
    st.stop()

# Load YOLO model
yolo_model = YOLO(YOLO_PATH)

# Upload image
st.header("Upload a PCB Image")
uploaded_img = st.file_uploader("Upload a .jpg/.jpeg/.png file", type=["jpg", "jpeg", "png"])

if uploaded_img:
    image = Image.open(uploaded_img).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Run YOLO prediction
    results = yolo_model.predict(source=image, conf=0.25)
    boxes = results[0].boxes

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    if boxes and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id] if yolo_model.names else "defect"

            draw.rectangle([(x1, y1), (x2, y2)], outline="white", width=3)
            draw.text((x1, y1 - 20), label, fill="white", font=font)

        st.image(image, caption="📍 Detected Defects", use_column_width=False)
    else:
        st.warning("⚠️ No defects detected.")
