import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logging

st.set_page_config(
    page_title="Age and Gender Estimation",
    page_icon=":Age:",
    initial_sidebar_state='collapsed'
)

st.title("Age and Gender Estimation")
st.write("A simple application to estimate the age and gender of a given face via cnn model.")
st.divider()

image_file = st.file_uploader("Please select your image.", type=["jpg", "png"])

with st.expander("Tips for improved accuracy"):
    st.write("Use images having more clarity")
    st.write("Use images with face having front view")
    st.write("Use images having enlarged face")
    st.write("Use colored images")

if image_file == None:
    st.write("Please select a valid image")
else:
    try:
        col1, col2 = st.columns([3,1])
        with col1:
            img = Image.open(image_file)
            st.image(img, width=480)
        with col2:
            prediction = PredictPipeline()
            with st.spinner("Estimating..."):
                results = prediction.predict(image_file)
            if results == -1:
                st.error("No face found !!!")
                st.write("Please select proper image.")
            elif results == 0:
                st.error("Multiple faces detected")
                st.write("Please select image containing one (1) face.")
            else:
                st.success('Extracted and predicted successfully!', icon="✅")
                age, gender = results['age'], results['gender']   
                st.write("Age: \t{}".format(age))
                st.write("Gender: \t{}".format(gender))
    except Exception as e:
        logging.info("Exception occured: {}".format(e))
        st.exception(e)