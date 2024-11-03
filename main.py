# import streamlit as st
# from transformers import pipeline
# from PIL import Image

# @st.cache_resource
# def load_pipeline():
#     return pipeline("image-classification", model="RickyIG/emotion_face_image_classification")

# pipe = load_pipeline()

# st.title("Emotion Detection from Face Image")

# uploadedImage = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploadedImage is not None:
#     image = Image.open(uploadedImage)
    
#     col1, col2 = st.columns(2)

#     with col2:
#         st.image(image, caption='Uploaded Image', use_column_width=True)

#     with col1:
#         predictions = pipe(image)
#         predicted_emotion = predictions[0]['label'].capitalize()

#         st.success(f"Predicted emotion: **{predicted_emotion}**")

import streamlit as st
from transformers import pipeline
from PIL import Image
import tempfile

@st.cache_resource
def load_classification_pipeline():
    return pipeline("image-classification", model="RickyIG/emotion_face_image_classification")

@st.cache_resource
def load_description_pipeline():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

classification_pipe = load_classification_pipeline()
description_pipe = load_description_pipeline()

st.title("Emotion Detection and Image Description")

uploadedImage = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploadedImage is not None:
    image = Image.open(uploadedImage)
    
    col1, col2 = st.columns(2)

    with col2:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    with col1:
        predictions = classification_pipe(image)
        predicted_emotion = predictions[0]['label'].capitalize()
        st.success(f"Predicted emotion: **{predicted_emotion}**")
        
        description = description_pipe(image)
        st.write(f"**Image description:** {description[0]['generated_text'].capitalize()}")