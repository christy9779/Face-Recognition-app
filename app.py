import cv2
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Face Recognition & Detection App", page_icon="👤", layout="wide"
)

st.title("👤 Face Detection & Recognition App")
st.write(
    "Upload an image below to detect and highlight human faces using OpenCV"
    " and computer vision algorithms."
)

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  # Read the image file bytes
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  img = cv2.imdecode(file_bytes, 1)

  # Convert image to grayscale for Haar Cascade detection
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Load OpenCV Haar Cascade Classifier using a safe fallback path
  cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
  face_cascade = cv2.CascadeClassifier(cascade_path)

  # Fallback if cv2.data fails in cloud environment
  if face_cascade.empty():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

  faces = face_cascade.detectMultiScale(
      gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
  )

  # Draw green bounding boxes around detected faces
  for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(
        img,
        "Face Detected",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

  # Convert OpenCV BGR format to RGB for correct Streamlit display
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # Layout Results
  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Processed Image Output")
    st.image(img_rgb, channels="RGB", use_container_width=True)

  with col2:
    st.subheader("Detection Analysis & Metrics")
    st.success(f"Successfully detected **{len(faces)}** face(s) in the image!")
    st.markdown("""
        * **Algorithm:** OpenCV Haar Cascade Classifier
        * **Preprocessing:** Grayscale conversion and scale-factor adjustment
        * **Performance:** Real-time bounding box rendering and classification
        """)
else:
  st.info("Please upload an image file to start face detection.")