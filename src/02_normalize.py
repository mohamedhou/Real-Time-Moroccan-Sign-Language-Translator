# src/02_normalize.py

import numpy as np
import glob
import os
from tqdm import tqdm

# ================== [ CONFIGURATION - KHTAR L'BI2A DYALK ] ==================
# 7iyed l'commentaire (uncomment) mn l'khyar li bghiti tkhdem bih.
# KHAS TKOUN NAFFS L'KHYAR LI KHDARITI F 01_preprocess.py

# --- KHAYAR 1: KHDMA F PC (LOKAL) ---
BASE_PATH = "."
PROCESSED_DATA_PATH = os.path.join(BASE_PATH, "data/processed_data")
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'train')
SCALER_FILE_PATH = os.path.join(BASE_PATH, "data/scaler.npz")
# ----------------------------------------

# --- KHAYAR 2: KHDMA F CLOUD (Google Colab / Kaggle) ---
# BASE_PATH = "/content/drive/MyDrive/Sign_Language_Project" # <-- Path dyalk f Google Drive
# PROCESSED_DATA_PATH = os.path.join(BASE_PATH, "data/processed_data")
# TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'train')
# SCALER_FILE_PATH = os.path.join(BASE_PATH, "data/scaler.npz")
# ----------------------------------------------------
# =========================================================================

def create_scaler(base_path):
    print("Chargement de toutes les données d'entraînement pour créer le scaler...")
    file_paths = sorted(glob.glob(os.path.join(base_path, "*", "*.npy")))
    
    if not file_paths:
        print(f"❌ ERREUR: Aucun fichier .npy trouvé dans : {base_path}")
        return

    all_sequences = []
    for f_path in tqdm(file_paths, desc="Chargement des fichiers"):
        all_sequences.append(np.load(f_path))
    
    full_data = np.array(all_sequences)
    
    print("\nCalcul du scaler...")
    num_sequences, seq_len, num_features = full_data.shape
    data_reshaped = full_data.reshape(-1, num_features)

    min_vals = data_reshaped.min(axis=0)
    max_vals = data_reshaped.max(axis=0)
    
    print(f"Sauvegarde des paramètres du scaler dans {SCALER_FILE_PATH}...")
    np.savez(SCALER_FILE_PATH, min_vals=min_vals, max_vals=max_vals)
    print("✅ Scaler créé avec succès ! Ce script a terminé sa tâche.")

if __name__ == "__main__":
    create_scaler(TRAIN_DATA_PATH)