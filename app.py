import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json, Sequential
from keras.saving import register_keras_serializable

@register_keras_serializable()
class MySequential(Sequential):
    pass

# Load model
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json, custom_objects={"Sequential": MySequential})
model.load_weights("facialemotionmodel.h5")

# Emotion labels
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# Helper
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Streamlit UI
st.title("Facial Emotion Detection App")
uploaded_file = st.file_uploader("Upload an image with a face", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        img = extract_features(face)
        pred = model.predict(img)
        emotion = labels[pred.argmax()]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    st.image(image, channels="BGR")
