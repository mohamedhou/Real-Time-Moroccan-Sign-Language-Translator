# app.py

import streamlit as st
import cv2
import numpy as np
import torch
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import json
import tempfile
import os

# Import l'architecture dyal l'modèle mn l'dossier src
from src.model_architecture import SignLanguageModel

# --- CONFIGURATION & MODEL LOADING ---
st.set_page_config(layout="wide")
st.title("🇲🇦 Traducteur de Langue des Signes Marocaine (LSM)")
st.write("Analysez une vidéo ou utilisez votre caméra en temps réel.")

# Paths (khasshom ykono mriglin)
MODEL_CHECKPOINT_PATH = "models/checkpoint.pth"
LABEL_MAP_PATH = "data/label_map.json"
SCALER_PATH = "data/scaler.npz" 
SEQUENCE_LENGTH = 150 # Mohim: khas tkon nafs l'9ima li sta3mlti f l'entraînement

# --- Fonctions Utilitaires ---
# Function l'loadé l'modèle (bach matb9ach t'loadé kol mrra)
@st.cache_resource
def load_all():
    try:
        # Load label map
        with open(LABEL_MAP_PATH, 'r') as f:
            label_map = json.load(f)
        num_classes = len(label_map)
        idx_to_class = {v: k for k, v in label_map.items()}

        # Load scaler
        scaler_data = np.load(SCALER_PATH)
        min_vals = scaler_data['min_vals']
        max_vals = scaler_data['max_vals']

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SignLanguageModel(input_size=1629, num_classes=num_classes).to(device)
        checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model, device, min_vals, max_vals, idx_to_class
    except FileNotFoundError as e:
        st.error(f"Erreur de chargement: Fichier manquant - {e.filename}")
        st.info("Assurez-vous que les fichiers 'checkpoint.pth', 'label_map.json', et 'scaler.npz' existent dans les bons dossiers ('models/' et 'data/').")
        return None, None, None, None, None

# Fonction dyal extraction
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


# Load a7ssan modèle 
model, device, min_vals, max_vals, idx_to_class = load_all()


# --- INTERFACE GRAPHIQUE (UI) ---
if model:
    # Kan9ado les TABS (onglets)
    tab1, tab2 = st.tabs(["🎥 Temps Réel (Caméra)", "📂 Analyser une Vidéo (Upload)"])

    # ========= Onglet 1: Caméra en Temps Réel =========
    with tab1:
        st.info("Cliquez sur 'START' pour activer votre caméra et commencer la traduction en direct.")
        
        class SignLanguageTransformer(VideoTransformerBase):
            def __init__(self):
                self.sequence = []
                self.predictions = []
                self.threshold = 0.8
                self.last_predicted_word = ""

            def transform(self, frame):
                image = frame.to_ndarray(format="bgr24")
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                
                keypoints = extract_keypoints(results)
                
                range_vals = max_vals - min_vals
                range_vals[range_vals == 0] = 1
                keypoints_normalized = 2 * (keypoints - min_vals) / range_vals - 1

                self.sequence.append(keypoints_normalized)
                self.sequence = self.sequence[-SEQUENCE_LENGTH:]

                if len(self.sequence) == SEQUENCE_LENGTH:
                    input_tensor = torch.tensor(np.expand_dims(self.sequence, axis=0), dtype=torch.float32).to(device)
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.softmax(output, dim=1)[0]
                        confidence, prediction_idx = torch.max(probabilities, dim=0)
                        
                        if confidence.item() > self.threshold:
                            predicted_word = idx_to_class[prediction_idx.item()]
                            if predicted_word != self.last_predicted_word:
                               self.last_predicted_word = predicted_word
                               self.predictions.append(predicted_word)
                               self.sequence = [] 

                # Affichage
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(self.predictions[-5:]), (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                return image

        webrtc_streamer(key="realtime", video_transformer_factory=SignLanguageTransformer,
                        media_stream_constraints={"video": True, "audio": False})

    # ========= Onglet 2: Analyse de Vidéo (Upload) =========
    with tab2:
        st.info("Téléchargez une vidéo d'un geste de la LSM et le modèle vous donnera sa signification.")
        uploaded_file = st.file_uploader("Choisissez un fichier vidéo (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])

        if uploaded_file is not None:
            st.video(uploaded_file) # N'affichiw l'video li t'uploadat
            
            with st.spinner("Analyse de la vidéo en cours..."):
                # 7it OpenCV kaytlb path, khass n'enregistréw l'video temporairement
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                
                cap = cv2.VideoCapture(tfile.name)
                video_frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)
                    keypoints = extract_keypoints(results)
                    video_frames.append(keypoints)
                cap.release()
                os.remove(tfile.name) # Nms7o l'fichier temporaire

                # Prédiction
                if len(video_frames) < SEQUENCE_LENGTH:
                    st.error(f"Erreur : La vidéo est trop courte. Elle doit contenir au moins {SEQUENCE_LENGTH} images (frames).")
                else:
                    # Akhod les keypoints dyal l'séquence
                    sequence_raw = np.array(video_frames[:SEQUENCE_LENGTH])
                    
                    # Normalization
                    range_vals = max_vals - min_vals
                    range_vals[range_vals == 0] = 1
                    sequence_normalized = 2 * (sequence_raw - min_vals) / range_vals - 1

                    input_tensor = torch.tensor(np.expand_dims(sequence_normalized, axis=0), dtype=torch.float32).to(device)

                    # Tawa9o3
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.softmax(output, dim=1)[0]
                        confidence, prediction_idx = torch.max(probabilities, dim=0)

                        predicted_word = idx_to_class[prediction_idx.item()]
                        conf_score = confidence.item() * 100

                        st.success(f"**Geste Détecté :** `{predicted_word.upper()}`")
                        st.metric(label="Confiance du Modèle", value=f"{conf_score:.2f} %")
                        st.progress(conf_score / 100)