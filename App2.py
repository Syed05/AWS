import streamlit as st
import torch
import pandas as pd
from PIL import Image

# Set up the Streamlit app header
st.title('YOLOv5 Object Detection')

# Add a file upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Add a control bar to select the confidence score
confidence_score = st.slider('Select the confidence score:', min_value=0.25, max_value=0.9, step=0.05, value=0.25)

# Display the input image after selecting it
if uploaded_file is not None:
    im = Image.open(uploaded_file)
    st.image(im, caption='Uploaded Image', use_column_width=True)

# Add a button to run the YOLOv5 model
if st.button('Detect objects') and uploaded_file is not None:
    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', './best.pt', force_reload=True)
    
    # Set the confidence score
    model.conf = confidence_score
    
    # Run object detection on the image
    results = model(im)
    
    # Display the detected objects in a table
    st.write('Detected objects:')
    st.dataframe(pd.DataFrame(results.pandas().xyxy[0]))
    
    # Display the resulting image
    st.subheader('Resulting image')
    im_result = results.render()
    st.image(im_result, caption='Detected Objects', use_column_width=True)
