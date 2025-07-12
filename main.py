"""
Training conv net to predict label.
1. Make data loader from data/rotated
2. Define conv net
3. Train conv net
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import os
import cv2

# --- Config ---
DATA_DIR = "data/rotated"
LABELS_PATH = "data/labels.csv"
NUM_CLASSES = 30
BATCH_SIZE = 16
EPOCHS = 16
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VideoDataset(Dataset):
    def __init__(self, data_dir, labels_path, transform=None):
        self.data_dir = data_dir
        self.labels = pd.read_csv(labels_path)
        self.transform = transform
        self.samples = []
        # Read all frames from all videos and store (frame, label) in self.samples
        for i in range(len(self.labels)):
            video_name = self.labels.iloc[i, 0]
            label = int(self.labels.iloc[i, 1])
            video_path = os.path.join(self.data_dir, video_name)
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (1024, 1024))
                self.samples.append((frame, label))
            cap.release()
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame, label = self.samples[idx]
        if self.transform:
            frame = self.transform(frame)
        else:
            frame = transforms.ToTensor()(frame)
        return frame, label


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    )
    print("Loading dataset...")
    dataset = VideoDataset(DATA_DIR, LABELS_PATH, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Dataset loaded with {len(dataset)} samples.")

    model = ConvNet(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    print(f"Initialized model {model}, optimizer {optimizer}, criterion {criterion}.")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for videos, labels in loader:
            videos = videos.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * videos.size(0)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(dataset):.4f}")

    torch.save(model.state_dict(), "weights/simple_convnet.pth")
    print("Training complete.")


def test():
    dataset = VideoDataset(DATA_DIR, LABELS_PATH, transform=None)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    for videos, labels in loader:
        print(f"Batch videos shape: {videos.shape}, Labels: {labels}")

if __name__ == "__main__":
    train()
    # test()
    