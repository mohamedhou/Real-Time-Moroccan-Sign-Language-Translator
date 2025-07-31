# src/01_preprocess.py

import os
import cv2
import numpy as np
import mediapipe as mp
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil

# ================== [ CONFIGURATION - KHTAR L'BI2A DYALK ] ==================
# 7iyed l'commentaire (uncomment) mn l'khyar li bghiti tkhdem bih.

# --- KHAYAR 1: KHDMA F PC (LOKAL) ---
BASE_PATH = "."
RAW_DATA_PATH = os.path.join(BASE_PATH, "data/raw_videos") # Dossier dyal les vidéos l'kham
PROCESSED_DATA_PATH = os.path.join(BASE_PATH, "data/processed_data") # Fin ghaymchiw les .npy
LABEL_MAP_PATH = os.path.join(BASE_PATH, "data/label_map.json")
# ----------------------------------------

# --- KHAYAR 2: KHDMA F CLOUD (Google Colab / Kaggle) ---
# DIMA ghadi tkon "Drive" f Colab w "/kaggle/working/" f Kaggle
# BASE_PATH = "/content/drive/MyDrive/Sign_Language_Project" # <-- Path dyalk f Google Drive
# RAW_DATA_PATH = os.path.join(BASE_PATH, "data/raw_videos") 
# PROCESSED_DATA_PATH = os.path.join(BASE_PATH, "data/processed_data")
# LABEL_MAP_PATH = os.path.join(BASE_PATH, "data/label_map.json")
# ----------------------------------------------------
# =========================================================================

# Paramètres
SEQUENCE_LENGTH = 150
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Setup MediaPipe
mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    # (L'code dyal had l'fonction kayb9a nafsso, ma tbdel walo hna)
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Main Logic
if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_PATH) or not any(os.scandir(RAW_DATA_PATH)):
        print(f"❌ ERREUR: Le dossier '{RAW_DATA_PATH}' est vide ou n'existe pas.")
        print("Veuillez y placer les dossiers des vidéos ('clavier', 'souris', etc.) avant de lancer ce script.")
    else:
        if os.path.exists(PROCESSED_DATA_PATH):
            shutil.rmtree(PROCESSED_DATA_PATH)
        os.makedirs(PROCESSED_DATA_PATH)
        print(f"Nettoyé et créé le dossier {PROCESSED_DATA_PATH}")

        # 1. Créer la map des labels
        actions = sorted([d for d in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, d))])
        label_map = {label: num for num, label in enumerate(actions)}
        with open(LABEL_MAP_PATH, 'w') as f:
            json.dump(label_map, f)
        print(f"Map des labels créée : {label_map}")

        # 2. Collecter tous les chemins des vidéos
        all_video_files = []
        for action_name in actions:
            action_path = os.path.join(RAW_DATA_PATH, action_name)
            for video_name in os.listdir(action_path):
                all_video_files.append((os.path.join(action_path, video_name), action_name))

        # 3. Diviser les données (train, val, test)
        train_val_files, test_files = train_test_split(all_video_files, test_size=TEST_SIZE, random_state=42, stratify=[f[1] for f in all_video_files])
        val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
        train_files, val_files = train_test_split(train_val_files, test_size=val_size_adjusted, random_state=42, stratify=[f[1] for f in train_val_files])

        datasets = {'train': train_files, 'val': val_files, 'test': test_files}

        # 4. Fonction principale de pré-traitement
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for set_name, file_list in datasets.items():
                print(f"\nTraitement du set '{set_name}'...")
                set_path = os.path.join(PROCESSED_DATA_PATH, set_name)
                os.makedirs(set_path)
                
                for video_path, action_name in tqdm(file_list, desc=f"Extraction des keypoints de {set_name}"):
                    dest_dir = os.path.join(set_path, action_name)
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    cap = cv2.VideoCapture(video_path)
                    video_frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret: break
                        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        keypoints = extract_keypoints(results)
                        video_frames.append(keypoints)
                    cap.release()

                    if len(video_frames) >= SEQUENCE_LENGTH:
                        sequence = np.array(video_frames[:SEQUENCE_LENGTH])
                        sequence_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}.npy"
                        np.save(os.path.join(dest_dir, sequence_filename), sequence)
        
        print("\n✅ Pré-traitement des données terminé !")