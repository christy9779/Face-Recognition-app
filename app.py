import urllib.request
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


# Robust Cascade Loader
@st.cache_resource
def load_face_cascade():
  xml_file = "haarcascade_frontalface_default.xml"
  try:
    path = cv2.data.haarcascades + xml_file
    cascade = cv2.CascadeClassifier(path)
    if not cascade.empty():
      return cascade
  except Exception:
    pass

  try:
    cascade = cv2.CascadeClassifier(xml_file)
    if not cascade.empty():
      return cascade
  except Exception:
    pass

  url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{xml_file}"
  urllib.request.urlretrieve(url, xml_file)
  return cv2.CascadeClassifier(xml_file)


face_cascade = load_face_cascade()

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  img = cv2.imdecode(file_bytes, 1)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = face_cascade.detectMultiScale(
      gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
  )

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

  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
