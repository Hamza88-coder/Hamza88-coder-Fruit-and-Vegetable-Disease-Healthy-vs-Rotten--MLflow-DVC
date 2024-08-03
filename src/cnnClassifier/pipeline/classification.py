import tensorflow as tf
import cv2
import json
import numpy as np
from tensorflow.keras.models import load_model

class ClassificationPipeline:
    def __init__(self, video_path):
        self.video_path = video_path

    def classifier(self, boxes_dict):
        print("i start classification")
        print(self.video_path)
        # Charger le modèle de classification
        classification_model = load_model('model/model.h5')
        
        cap = cv2.VideoCapture(self.video_path)
        # Vérifier si la vidéo a été correctement chargée
        if not cap.isOpened():
            print("Erreur : Impossible de charger la vidéo.")
            exit()

        # Obtenir les propriétés de la vidéo
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Initialiser l'écriture vidéo
        output_path = 'videos/classification_output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0

        while cap.isOpened():
            print("am in while")
            ret, frame = cap.read()
            if not ret:
                print("i will reak")
                break

            # Obtenir les boîtes pour la frame courante
            frame_boxes = boxes_dict.get(frame_idx, [])
            print(frame_boxes)
            for (x1, y1, x2, y2, score) in frame_boxes:
                print("fadwa")
                if x2 >= 1500 and x2 < 1900:
                    # Extraire la région d'intérêt (ROI) pour la banane détectée
                    banana_roi = frame[y1:y2, x1:x2]
                    
                    # Prétraiter la ROI pour le modèle de classification
                    input_size = (224, 224)  # Taille d'entrée attendue par le modèle
                    banana_resized = cv2.resize(banana_roi, input_size)
                    banana_normalized = banana_resized / 255.0
                    banana_batch = np.expand_dims(banana_normalized, axis=0)  # Ajouter une dimension pour le batch
                    
                    # Effectuer la prédiction
                    predictions = classification_model.predict(banana_batch)
                    
                    predicted_class = np.argmax(predictions, axis=1)[0]
                    # Déterminer l'étiquette de classe prédite
                    if predicted_class == 1 and predictions[0][1] > 0.7:
                        label = "Healthy"
                        color = (0, 255, 0)  # Vert pour healthy
                    elif predicted_class == 4 and predictions[0][4] > 0.7: 
                        label = "Rotten"
                        color = (0, 0, 255)  # Rouge pour rotten
                    else:
                        label = "not yet"
                        color = (0, 0, 0)
                    
                    # Annoter la frame avec les boîtes et les étiquettes
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label}: {score:.2f}', (x1, y1 + y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

            # Écrire la frame annotée dans le fichier vidéo
            out.write(frame)
            frame_idx += 1
        
        # Libérer les ressources
        cap.release()
        out.release()
        print("i finished my programme")
