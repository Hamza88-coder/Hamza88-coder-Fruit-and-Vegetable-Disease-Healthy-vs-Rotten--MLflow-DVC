
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import json
from ultralytics import YOLO
import pickle
import os

class PredictionPipeline:
  def __init__(self,video_path):
        self.video_path =video_path


  def predict(self,target_class_id=46):
  
    # Load the YOLOv8 model
    model = YOLO("yolov8x.pt")

    # Load the video
    cap = cv2.VideoCapture(self.video_path)

    # Check if the video was successfully loaded
    if not cap.isOpened():
        print("Error: Unable to load the video.")
        return

    # Dictionary to store detected boxes
    boxes_dict = {}

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply YOLO to each frame
        results = model(frame)

        # Process the results and save the boxes
        frame_boxes = []
        for result in results:
            bboxes = result.boxes.xyxy.cpu().numpy().tolist() # Box coordinates
            scores = result.boxes.conf.cpu().numpy().tolist()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy().tolist()  # Class IDs

            for bbox, score, class_id in zip(bboxes, scores, class_ids):
                if class_id == target_class_id:
                    x1, y1, x2, y2 = map(int, bbox)
                    frame_boxes.append((x1, y1, x2, y2, score))

        # Save the boxes for the current frame
        boxes_dict[frame_idx] = frame_boxes
        frame_idx += 1

    # Release the resources
    cap.release()

    # Enregistrer le dictionnaire dans un fichier JSON
    #with open('dict_banae.json', 'w') as f:
        #boxes_dict= json.dump(boxes_dict, f, indent=4)  # Utiliser indent pour une meilleure lisibilité
        

    print(boxes_dict)
    return boxes_dict




    
    
        
    
      
        

   



