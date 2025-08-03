# src/03_train.py

# ==============================================================================
# PARTIE 1: IMPORTS - KANJIBO L'3OTLA (LES OUTILS)
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import json
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

#  importé l'architecture dyal l'modèle
from model_architecture import SignLanguageModel

# ================== [ CONFIGURATION - KHTAR L'BI2A DYALK ] ==================
# 7iyed l'commentaire (uncomment) mn l'khyar li bghiti tkhdem bih.
# KHAS TKOUN NAFFS L'KHYAR LI KHDARITI F 01_preprocess.py

# --- KHAYAR 1: KHDMA F PC (LOKAL) ---
BASE_PATH = "."
DATA_PATH = os.path.join(BASE_PATH, "data/processed_data")
LABEL_MAP_PATH = os.path.join(BASE_PATH, "data/label_map.json")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "models/checkpoint.pth")
HISTORY_SAVE_PATH = os.path.join(BASE_PATH, "models/training_history.json")
# ----------------------------------------

# --- KHAYAR 2: KHDMA F CLOUD (Google Colab / Kaggle) ---
# BASE_PATH = "/content/drive/MyDrive/Sign_Language_Project" # <-- Path dyalk f Google Drive
# DATA_PATH = os.path.join(BASE_PATH, "data/processed_data")
# LABEL_MAP_PATH = os.path.join(BASE_PATH, "data/label_map.json")
# MODEL_SAVE_PATH = os.path.join(BASE_PATH, "models/checkpoint.pth")
# HISTORY_SAVE_PATH = os.path.join(BASE_PATH, "models/training_history.json")
# ----------------------------------------------------
# =========================================================================

# Paramètres d'entraînement
BATCH_SIZE = 32
NUM_EPOCHS = 150
LEARNING_RATE = 0.0001
INPUT_FEATURES = 1629
PATIENCE = 20 # Sber dyal Early Stopping

# ==============================================================================
# PARTIE 3: DEVICE - KAN3ZLO L'MAQUINA (GPU/CPU)
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==============================================================================
# PARTIE 4: DATASET CLASS - L'MAGASIN DYAL LES DONNÉES
# ==============================================================================
class SignLanguageKeypointDataset(Dataset):
    def __init__(self, data_path, label_map):
        self.sequences = []
        for action_name, label in label_map.items():
            action_dir = os.path.join(data_path, action_name)
            if not os.path.isdir(action_dir):
                print(f"Warning: Directory not found for action '{action_name}'")
                continue
            for seq_file in glob.glob(os.path.join(action_dir, '*.npy')):
                self.sequences.append((seq_file, label))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_path, label = self.sequences[idx]
        keypoints = np.load(seq_path)
        return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ==============================================================================
# PARTIE 5: FONCTIONS DYAL CHECKPOINT
# ==============================================================================
def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    start_epoch = 0
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    if os.path.exists(checkpoint_path):
        print(f"=> Loading checkpoint from '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        history = checkpoint.get('history', history)
    else:
        print("=> No checkpoint found, starting from scratch")
    return start_epoch, best_val_acc, history

# ==============================================================================
# PARTIE 6: FONCTION DYAL L'ENTRAÎNEMENT
# ==============================================================================
def train_model():
    # --- A. Setup & DataLoading ---
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
    NUM_CLASSES = len(label_map)

    # Kan'crééw les datasets (train, val)
    train_dataset = SignLanguageKeypointDataset(os.path.join(DATA_PATH, 'train'), label_map)
    val_dataset = SignLanguageKeypointDataset(os.path.join(DATA_PATH, 'val'), label_map)
    
    # Kan'crééw les DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Found {len(train_dataset)} training sequences.")
    print(f"Found {len(val_dataset)} validation sequences.")

    # Kanbniw l'modèle
    model = SignLanguageModel(input_size=INPUT_FEATURES, num_classes=NUM_CLASSES).to(DEVICE)
    
    # Optimizer, Scheduler, w Loss Function
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, min_lr=1e-6, verbose=True)
    criterion = nn.CrossEntropyLoss()

    # --- B. Load Checkpoint (ila kan) ---
    start_epoch, best_val_acc, history = load_checkpoint(MODEL_SAVE_PATH, model, optimizer, scheduler)
    epochs_no_improve = 0

    # --- C. Boucle dyal l'Entraînement ---
    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # --- Validation Phase ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # --- Calculate and store metrics ---
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = train_correct / train_total
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = val_correct / val_total
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        scheduler.step(epoch_val_acc)

        # --- Checkpointing & Early Stopping ---
        if epoch_val_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.4f} to {epoch_val_acc:.4f}. Saving model...")
            best_val_acc = epoch_val_acc
            epochs_no_improve = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }
            save_checkpoint(checkpoint, MODEL_SAVE_PATH)
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= PATIENCE:
            print(f"🛑 Early stopping triggered after {PATIENCE} epochs without improvement.")
            break
            
    print("🏁 Training finished!")
    
    # Save training history l fichier JSON
    os.makedirs(os.path.dirname(HISTORY_SAVE_PATH), exist_ok=True)
    with open(HISTORY_SAVE_PATH, 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {HISTORY_SAVE_PATH}")
    return history

# ==============================================================================
# PARTIE 7: FONCTION DYAL L'AFFICHAGE DYAL L'RÉSULTATS
# ==============================================================================
def plot_history(history):
    epochs_range = range(len(history['train_acc']))
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.savefig(os.path.join(os.path.dirname(MODEL_SAVE_PATH), "training_curves.png"))
    print(f"Training curves saved to 'models/training_curves.png'")
    # F Colab, plt.show() ma katkhedmch dima mzyan f script.
    # L'afdal howa t'sauvegardé l'image w t'afficherha men be3d.
    #plt.show() 

# ==============================================================================
# PARTIE 8: L'EXÉCUTION PRINCIPALE
# ==============================================================================
if __name__ == "__main__":
    try:
        # Lancer l'entraînement
        training_history = train_model()
        
        # Mli yssali, afficher les courbes
        if training_history:
            print("\nPlotting training history...")
            plot_history(training_history)
    except Exception as e:
        print(f"\nAn error occurred during the training script: {e}")   