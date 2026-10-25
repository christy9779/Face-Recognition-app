import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Face Recognition & Detection App", page_icon="👤", layout="wide"
)

st.title("👤 Face Detection & Recognition App")
st.write(
    "Upload an image below to detect and highlight human faces using MediaPipe"
    " and computer vision algorithms."
)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  # Read the image file bytes
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  img = cv2.imdecode(file_bytes, 1)
  H, W, _ = img.shape

  # Convert BGR to RGB for MediaPipe processing
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  face_count = 0
  with mp_face_detection.FaceDetection(
      model_selection=1, min_detection_confidence=0.5
  ) as face_detection:
    results = face_detection.process(img_rgb)

    if results.detections:
      face_count = len(results.detections)
      for detection in results.detections:
        # Extract bounding box coordinates
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * W)
        y = int(bbox.ymin * H)
        w = int(bbox.width * W)
        h = int(bbox.height * H)

        # Draw bounding box and label
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(
            img,
            "Face Detected",
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

  # Convert back to RGB for Streamlit display
  img_final = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # Layout Results
  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Processed Image Output")
    st.image(img_final, channels="RGB", use_container_width=True)

  with col2:
    st.subheader("Detection Analysis & Metrics")
    st.success(f"Successfully detected **{face_count}** face(s) in the image!")
    st.markdown("""
        * **Algorithm:** MediaPipe Face Detection (BlazeFace Neural Network)
        * **Preprocessing:** RGB color space conversion & thresholding
        * **Performance:** High-accuracy bounding box rendering
        """)
else:
  st.info("Please upload an image file to start face detection.")
