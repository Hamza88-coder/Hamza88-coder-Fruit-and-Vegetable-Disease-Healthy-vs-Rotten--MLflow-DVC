import streamlit as st
import os
import tempfile
from src.cnnClassifier.pipeline.prediction import PredictionPipeline
from src.cnnClassifier.utils.common import read_video
from src.cnnClassifier.pipeline.classification import ClassificationPipeline

# Ensure environment variables are set
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

class ClientApp:
    def __init__(self):
        self.video_path = None
        self.classifier = None
        self.dict_prediction = None

clApp = ClientApp()

st.title("Healthy_VS_Rotten")

# Display an image
image_path = "class_image.jpg"  # Replace with your image path
st.image(image_path, caption="Sample Image", use_column_width=True)

# Training Section
st.header("Train Model")

if st.button("Train"):
    try:
        os.system("python main.py")
        # os.system("dvc repro")
        st.success("Training done successfully!")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Prediction Section
st.header("Predict with Uploaded Video")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4"])

if uploaded_file:
    # Create a temporary file to store the uploaded video
    if uploaded_file is not None:
        # Ensure the temporary directory exists
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        # Save uploaded file to temporary directory
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        # Read video frames
        st.sidebar.text(f"Video loaded: {uploaded_file.name}")

    # Update the path in ClientApp and process the video
    prediction = PredictionPipeline(temp_file_path)
    clApp.dict_prediction = prediction.predict()
    classification = ClassificationPipeline(temp_file_path)
    classification.classifier(clApp.dict_prediction)
    video_file = "videos/classification_output.mp4"  # Remplacez par le chemin de votre vidéo

# Afficher la vidéo
    st.video(video_file)

    # Clean up: Delete the temporary file with a delay
    import time
    time.sleep(1)  # Delay to ensure the file is not being used
    if os.path.exists(temp_file_path):
        try:
            os.remove(temp_file_path)
        except Exception as e:
            st.error(f"Error deleting temporary file: {e}")
