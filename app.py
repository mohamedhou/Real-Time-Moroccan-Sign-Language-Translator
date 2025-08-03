# app.py (Version 3.0 - Professional UI)

import streamlit as st
import cv2
import numpy as np
import torch
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import json
import tempfile
import os

# Import l'architecture
from src.model_architecture import SignLanguageModel

# ==================== CSS INJECTION ====================
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style/style.css")

# ==================== CONFIGURATION ====================
st.title("🇲🇦 Traducteur de Langue des Signes Marocaine (LSM)")

# Paths
MODEL_CHECKPOINT_PATH = "models/checkpoint.pth"
LABEL_MAP_PATH = "data/label_map.json"
SCALER_PATH = "data/scaler.npz"
SEQUENCE_LENGTH = 150

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_all():
    # ... (Had l'fonction ma katbedelch) ...
    try:
        with open(LABEL_MAP_PATH, 'r') as f:
            label_map = json.load(f)
        num_classes = len(label_map)
        idx_to_class = {v: k for k, v in label_map.items()}
        scaler_data = np.load(SCALER_PATH)
        min_vals, max_vals = scaler_data['min_vals'], scaler_data['max_vals']
        device = torch.device("cpu")
        model = SignLanguageModel(input_size=1629, num_classes=num_classes).to(device)
        checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, device, min_vals, max_vals, idx_to_class
    except FileNotFoundError:
        return None, None, None, None, None

# ... (L'ba9i dyal les fonctions dyal extraction kayb9a nafsso) ...
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

model, device, min_vals, max_vals, idx_to_class = load_all()


# ==================== SIDEBAR ====================
st.sidebar.markdown("<h2>📖 À Propos du Projet</h2>", unsafe_allow_html=True)
st.sidebar.info(
    "Cette application utilise un modèle de Deep Learning (CNN + LSTM) pour interpréter "
    "la Langue des Signes Marocaine en temps réel."
)
# BDEL HAD LES LIENS
st.sidebar.markdown("---")
st.sidebar.markdown("**Auteur:** Mohamed Houcht")
st.sidebar.markdown("**GitHub Repo:** [Lien vers votre projet](https://github.com/mohamedhou/Real-Time-Moroccan-Sign-Language-Translator)")
st.sidebar.markdown("**LinkedIn:** [Votre Profil](https://linkedin.com/in/mohamed-houcht-861494289)")

# ==================== MAIN UI ====================
if model is None:
    st.error(f"**Erreur de chargement :** Un ou plusieurs fichiers sont manquants.")
    st.warning("Veuillez vous assurer que le modèle est entraîné et que les fichiers `checkpoint.pth`, `label_map.json`, et `scaler.npz` sont présents dans les dossiers `models/` et `data/`.")
else:
    tab1, tab2 = st.tabs(["🎥 Traduction en Temps Réel", "📂 Analyser une Vidéo"])

    # --- TAB 1: Real-Time ---
    with tab1:
        # (L'code dyal l'classe SignLanguageTransformer kayb9a nafffsso, ma tbdel fih walo)
        class SignLanguageTransformer(VideoTransformerBase):
            def __init__(self):
                self.sequence, self.predictions, self.last_predicted_word = [], [], ""
                self.threshold = 0.8
            def transform(self, frame):
                image = frame.to_ndarray(format="bgr24")
                results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                keypoints = extract_keypoints(results)
                range_vals = max_vals - min_vals; range_vals[range_vals == 0] = 1
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
                                self.last_predicted_word, self.predictions = predicted_word, self.predictions + [predicted_word]
                                self.sequence = []
                # Affichage
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(self.predictions[-5:]), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                return image
        
        webrtc_streamer(key="realtime", video_transformer_factory=SignLanguageTransformer)

    # --- TAB 2: Video Upload ---
    with tab2:
        col1, col2 = st.columns([2, 1]) # Column lwla kbira 3la tanya
        
        with col1:
            uploaded_file = st.file_uploader("Choisissez une vidéo...", type=["mp4", "mov", "avi"])
            if uploaded_file:
                st.video(uploaded_file)
        
        with col2:
            st.markdown("### Résultat de l'analyse")
            if uploaded_file:
                with st.spinner("Analyse en cours..."):
                    # (L'code dyal l'traitement dyal l'vidéo kayb9a naffso)
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4'); tfile.write(uploaded_file.read())
                    cap = cv2.VideoCapture(tfile.name); video_frames = []
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        video_frames.append(extract_keypoints(results))
                    cap.release(); os.remove(tfile.name)
                    
                    if len(video_frames) < SEQUENCE_LENGTH:
                        st.error(f"Vidéo trop courte (< {SEQUENCE_LENGTH} frames)")
                    else:
                        sequence_raw = np.array(video_frames[:SEQUENCE_LENGTH])
                        range_vals = max_vals - min_vals; range_vals[range_vals == 0] = 1
                        sequence_normalized = 2 * (sequence_raw - min_vals) / range_vals - 1
                        input_tensor = torch.tensor(np.expand_dims(sequence_normalized, axis=0), dtype=torch.float32).to(device)
                        
                        with torch.no_grad():
                            output = model(input_tensor)
                            probabilities = torch.softmax(output, dim=1)[0]
                            confidence, prediction_idx = torch.max(probabilities, dim=0)
                            predicted_word, conf_score = idx_to_class[prediction_idx.item()], confidence.item()

                        # HTML Custom l l'affichage
                        st.markdown(f"""
                        <div class="result-container">
                            <div class="label">Geste Détecté</div>
                            <div class="prediction">{predicted_word.upper()}</div>
                            <br>
                        </div>
                        """, unsafe_allow_html=True)
                        st.metric(label="Confiance du Modèle", value=f"{conf_score*100:.2f} %")
            else:
                st.info("En attente d'une vidéo à analyser.")