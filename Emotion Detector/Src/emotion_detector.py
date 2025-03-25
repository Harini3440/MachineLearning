import numpy as np
import librosa
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm  # For progress bars

def extract_audio_features(file_path, sample_rate=22050):
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate)
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr), axis=1)
        features = np.concatenate([mfcc, chroma, mel])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

class AudioEmotionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class EmotionNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(EmotionNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Add dropout for regularization
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)  # No softmax here; handled by loss function
        return x


def prepare_dataset(data_dir):
    features_list = []
    labels = []
    emotion_dict = {'happy': 0, 'sad': 1, 'angry': 2, 'neutral': 3}
    
    for emotion in os.listdir(data_dir):
        emotion_path = os.path.join(data_dir, emotion)
        if os.path.isdir(emotion_path):
            audio_files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
            for audio_file in tqdm(audio_files, desc=f"Processing {emotion}"):  # Progress bar
                file_path = os.path.join(emotion_path, audio_file)
                features = extract_audio_features(file_path)
                if features is not None:
                    features_list.append(features)
                    labels.append(emotion_dict.get(emotion, -1))
    
    X = np.array(features_list)
    y = np.array(labels)
    
   
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, "models/scaler.pkl")  # Save scaler to models/
    
    return X, y

def train_emotion_detector(data_dir, epochs=50, batch_size=32):
    # Prepare data
    X, y = prepare_dataset(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    train_dataset = AudioEmotionDataset(X_train, y_train)
    test_dataset = AudioEmotionDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
        input_size = X.shape[1]  # Number of features (13 MFCC + 12 chroma + 128 mel = 153)
    hidden_size1 = 256
    hidden_size2 = 128
    num_classes = len(np.unique(y))  # Number of emotions
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionNet(input_size, hidden_size1, hidden_size2, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    
    best_accuracy = 0
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Test Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "models/emotion_net.pth")
    
    return model

# Prediction function
def predict_emotion(model, audio_file):
    model.eval()
    features = extract_audio_features(audio_file)
    if features is None:
        return "Error in feature extraction"
    
    # Load scaler and normalize features
    scaler = joblib.load("models/scaler.pkl")
    features = scaler.transform([features])  # Add batch dimension for scaler
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension for model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = features.to(device)
    
    with torch.no_grad():
        output = model(features)
        _, predicted = torch.max(output, 1)
    
    emotion_dict = {0: 'happy', 1: 'sad', 2: 'angry', 3: 'neutral'}
    return emotion_dict.get(predicted.item(), "Unknown")


if __name__ == "__main__":
    # Use Windows-style or relative path
    dataset_path = "data"  # Assumes data/ is in the project root
    # Ensure models/ exists
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Train the model
    emotion_model = train_emotion_detector(dataset_path, epochs=50, batch_size=32)
    
    # Test with a sample audio file
    test_audio = "data\\happy\\happy1.wav"  
    predicted_emotion = predict_emotion(emotion_model, test_audio)
    print(f"Predicted Emotion: {predicted_emotion}")