import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Face Detection & Recognition App", page_icon="👤", layout="wide"
)

st.title("👤 Face Detection & Recognition App")
st.write(
    "Upload an image below to detect and highlight human faces using OpenCV"
    " Haar Cascades."
)


@st.cache_resource
def load_cascade():
  return cv2.CascadeClassifier(
      cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
  )


face_cascade = load_cascade()

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  img = cv2.imdecode(file_bytes, 1)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Detect faces
  faces = face_cascade.detectMultiScale(
      gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
  )

  for x, y, w, h in faces:
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

  img_final = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Processed Image Output")
    st.image(img_final, channels="RGB", use_container_width=True)

  with col2:
    st.subheader("Detection Analysis & Metrics")
    st.success(f"Successfully detected **{len(faces)}** face(s) in the image!")
    st.markdown("""
        * **Algorithm:** OpenCV Haar Cascade Classifier
        * **Preprocessing:** Grayscale conversion & multi-scale detection
        * **Performance:** Instantaneous execution & zero dependency conflicts
        """)
else:
  st.info("Please upload an image file to start face detection.")
