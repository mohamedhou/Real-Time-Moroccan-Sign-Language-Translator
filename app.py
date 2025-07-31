import streamlit as st
import cv2
import numpy as np
import torch
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import json

# Import l'architecture dyal l'modèle mn l'dossier src
from src.model_architecture import SignLanguageModel

# --- Configuration & Model Loading ---
st.set_page_config(layout="wide")
st.title("🇲🇦 Traducteur de Langue des Signes Marocaine en Temps Réel")
st.write("Cette application utilise un modèle d'IA pour traduire les gestes de la LSM en texte.")

# Paths (khasshom ykono mriglin)
MODEL_CHECKPOINT_PATH = "models/checkpoint.pth"
LABEL_MAP_PATH = "data/label_map.json"
SCALER_PATH = "data/scaler.npz" # Fichier dyal l'normalization

# Function l'loadé l'modèle (bach matb9ach t'loadé kol mrra)
@st.cache_resource
def load_model():
    try:
        with open(LABEL_MAP_PATH, 'r') as f:
            label_map = json.load(f)
        num_classes = len(label_map)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = SignLanguageModel(input_size=1629, num_classes=num_classes).to(device)
        
        checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Load scaler
        scaler_data = np.load(SCALER_PATH)
        min_vals = scaler_data['min_vals']
        max_vals = scaler_data['max_vals']

        # Reverse map (mn ra9m l ism)
        idx_to_class = {v: k for k, v in label_map.items()}

        return model, device, min_vals, max_vals, idx_to_class
    except FileNotFoundError:
        return None, None, None, None, None

model, device, min_vals, max_vals, idx_to_class = load_model()

# Setup MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Real-Time Translation Logic ---
if model is None:
    st.error("Erreur: Le fichier du modèle (`checkpoint.pth`), `label_map.json` ou `scaler.npz` est introuvable.")
    st.info("Veuillez vous assurer que le modèle est entraîné et que les fichiers sont placés dans les bons dossiers (`models/` et `data/`).")
else:
    class SignLanguageTransformer(VideoTransformerBase):
        def __init__(self):
            self.sequence = []
            self.predictions = []
            self.threshold = 0.8 # L'ihtimal l'adna bach n'affichiw l'natija
            self.last_predicted_word = ""

        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")
            
            # MediaPipe processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            # Keypoint extraction
            pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
            face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
            keypoints = np.concatenate([pose, face, lh, rh])
            
            # Normalization
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1
            keypoints_normalized = 2 * (keypoints - min_vals) / range_vals - 1

            self.sequence.append(keypoints_normalized)
            self.sequence = self.sequence[-150:] # Kan7afdo ghir 3la l'kher 150 frame

            # Prediction
            if len(self.sequence) == 150:
                input_tensor = torch.tensor(np.expand_dims(self.sequence, axis=0), dtype=torch.float32).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)[0]
                    prediction_idx = torch.argmax(probabilities).item()
                    confidence = probabilities[prediction_idx].item()
                    
                    if confidence > self.threshold:
                        predicted_word = idx_to_class[prediction_idx]
                        if predicted_word != self.last_predicted_word:
                           self.last_predicted_word = predicted_word
                           self.predictions.append(predicted_word)
                           # Reset sequence to avoid rapid-fire predictions
                           self.sequence = [] 

            # Displaying the result on the frame
            # (Hadi l'partie dyal l'affichage fo9 l'video)
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(self.predictions[-5:]), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            return image

    webrtc_streamer(key="sign-language", video_transformer_factory=SignLanguageTransformer)