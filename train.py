#!pip install pydicom torch torchvision pandas numpy opencv-python
#uncomment for google co lab

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import cv2
import pydicom
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')
base_path = '/content/drive/MyDrive/Cropped Dataset/'

# List subdirectories
subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
print("Found subdirectories:", subdirs)

# Convert DICOM to PNG
def dicom_to_png(dicom_path, output_dir):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    output_path = os.path.join(output_dir, os.path.basename(dicom_path).replace('.dcm', '.png'))
    cv2.imwrite(output_path, img)
    return output_path

dicom_paths = []
for subdir in subdirs:
    full_path = os.path.join(base_path, subdir)
    dicom_files = [os.path.join(full_path, f) for f in os.listdir(full_path) if f.endswith('.dcm')]
    dicom_paths.extend(dicom_files)
print(f"Found {len(dicom_paths)} DICOM files")

output_base = '/content/converted_images/'
os.makedirs(output_base, exist_ok=True)
converted_paths = []
for dicom_path in dicom_paths[:1000]:  # Adjust limit as needed
    png_path = dicom_to_png(dicom_path, output_base)
    converted_paths.append(png_path)

# Parse labels from filenames
df = pd.DataFrame({'image file path': converted_paths})
df['label'] = df['image file path'].apply(
    lambda x: 1 if 'MALIGNANT' in os.path.basename(x).upper() else 0
)
print("Sample paths and labels:")
print(df.head())
print("Class distribution:", df['label'].value_counts())

# Dataset class
class CroppedDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    def __len__(self): return len(self.dataframe)
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image file path']
        label = self.dataframe.iloc[idx]['label']
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# Transforms with augmentation
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataloaders
train_size = int(0.8 * len(df))
val_size = len(df) - train_size
train_df, val_df = df[:train_size], df[train_size:]

train_dataset = CroppedDataset(train_df, transform=data_transforms)
val_dataset = CroppedDataset(val_df, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR

# Enhanced training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

# Train
print("Starting training...")
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# Save model
torch.save(model.state_dict(), '/content/resnet18_cropped.pth')
print("Model saved")


#Expected output on google co lab

'''
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
Found subdirectories: ['Calc-Training-Final', 'Mass-Test-Final', 'Mass-Training-Final', 'Calc-Test-Final']
Found 3568 DICOM files
Sample paths and labels:
                                     image file path  label
0  /content/converted_images/P_02584_cropped_BENI...      0
1  /content/converted_images/P_00680_cropped_BENI...      0
2  /content/converted_images/P_00680_cropped_BENI...      0
3  /content/converted_images/P_00680_cropped_BENI...      0
4  /content/converted_images/P_00680_cropped_BENI...      0
Class distribution: label
0    622
1    378
Name: count, dtype: int64
Starting training...
Epoch 1/10:
Train Loss: 0.6348, Train Accuracy: 65.12%
Val Loss: 0.5838, Val Accuracy: 71.50%
Epoch 2/10:
Train Loss: 0.4982, Train Accuracy: 73.38%
Val Loss: 0.5654, Val Accuracy: 74.50%
Epoch 3/10:
Train Loss: 0.4317, Train Accuracy: 79.88%
Val Loss: 0.5516, Val Accuracy: 74.50%
Epoch 4/10:
Train Loss: 0.3719, Train Accuracy: 82.88%
Val Loss: 0.6508, Val Accuracy: 71.00%
Epoch 5/10:
Train Loss: 0.3057, Train Accuracy: 87.12%
Val Loss: 0.5222, Val Accuracy: 77.00%
Epoch 6/10:
Train Loss: 0.2810, Train Accuracy: 88.75%
Val Loss: 0.7624, Val Accuracy: 67.00%
Epoch 7/10:
Train Loss: 0.2473, Train Accuracy: 90.25%
Val Loss: 0.6676, Val Accuracy: 73.50%
Epoch 8/10:
Train Loss: 0.1985, Train Accuracy: 92.38%
Val Loss: 0.6183, Val Accuracy: 72.50%
Epoch 9/10:
Train Loss: 0.1739, Train Accuracy: 93.12%
Val Loss: 0.9341, Val Accuracy: 62.50%
Epoch 10/10:
Train Loss: 0.1747, Train Accuracy: 93.88%
Val Loss: 0.8051, Val Accuracy: 74.50%
Model saved

'''