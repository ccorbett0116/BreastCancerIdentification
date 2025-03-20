# Install dependencies
# !pip install torch torchvision pandas numpy opencv-python
# uncomment ^ for google co lab

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')
png_base_path = '/content/drive/MyDrive/Cropped Dataset/Converted PNGs Structured/'

# Hyperparameters
epochs = 10
learning_rate = 0.0001 
batch_size = 64
dropout_rate = 0.5
class_weights = torch.tensor([1.0, 2.0])  # Increased Malignant weight

# Define subdirectories
train_subdirs = ['Calc-Training-Final', 'Mass-Training-Final']
test_subdirs = ['Calc-Test-Final', 'Mass-Test-Final']

# Load precomputed PNGs
train_converted_paths = []
for subdir in train_subdirs:
    subdir_path = os.path.join(png_base_path, subdir)
    if os.path.exists(subdir_path):
        train_converted_paths.extend([os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.png')])
    else:
        print(f"Warning: {subdir_path} not found")

test_converted_paths = []
for subdir in test_subdirs:
    subdir_path = os.path.join(png_base_path, subdir)
    if os.path.exists(subdir_path):
        test_converted_paths.extend([os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.png')])
    else:
        print(f"Warning: {subdir_path} not found")

print(f"Found {len(train_converted_paths)} training PNG files")
print(f"Found {len(test_converted_paths)} validation PNG files")

# Parse labels
train_df = pd.DataFrame({'image file path': train_converted_paths})
train_df['label'] = train_df['image file path'].apply(
    lambda x: 1 if 'MALIGNANT' in os.path.basename(x).upper() else 0
)
print("Training sample paths and labels:")
print(train_df.head())
print("Training class distribution:", train_df['label'].value_counts())

test_df = pd.DataFrame({'image file path': test_converted_paths})
test_df['label'] = test_df['image file path'].apply(
    lambda x: 1 if 'MALIGNANT' in os.path.basename(x).upper() else 0
)
print("Test (validation) sample paths and labels:")
print(test_df.head())
print("Test (validation) class distribution:", test_df['label'].value_counts())

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

# Transforms
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add translation
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataloaders
train_dataset = CroppedDataset(train_df, transform=train_transforms)
val_dataset = CroppedDataset(test_df, transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Model with adjusted freezing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)
model = models.resnet50(weights='IMAGENET1K_V1')
# for name, param in model.named_parameters():
#     if 'layer3' not in name and 'layer4' not in name and 'fc' not in name: 
#         param.requires_grad = False
model.fc = nn.Sequential(
    nn.Dropout(dropout_rate),
    nn.Linear(model.fc.in_features, 2)
)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    for epoch in range(epochs):
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

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        scheduler.step(val_loss)

# Train
print("Starting training...")
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs)

# Save model
torch.save(model.state_dict(), '/content/resnet50_cropped.pth')
print("Model saved")

'''
Found 2864 training PNG files
Found 704 validation PNG files
Training sample paths and labels:
                                     image file path  label
0  /content/drive/MyDrive/Cropped Dataset/Convert...      0
1  /content/drive/MyDrive/Cropped Dataset/Convert...      0
2  /content/drive/MyDrive/Cropped Dataset/Convert...      0
3  /content/drive/MyDrive/Cropped Dataset/Convert...      0
4  /content/drive/MyDrive/Cropped Dataset/Convert...      0
Training class distribution: label
0    1683
1    1181
Name: count, dtype: int64
Test (validation) sample paths and labels:
                                     image file path  label
0  /content/drive/MyDrive/Cropped Dataset/Convert...      0
1  /content/drive/MyDrive/Cropped Dataset/Convert...      0
2  /content/drive/MyDrive/Cropped Dataset/Convert...      0
3  /content/drive/MyDrive/Cropped Dataset/Convert...      0
4  /content/drive/MyDrive/Cropped Dataset/Convert...      0
Test (validation) class distribution: label
0    428
1    276
Name: count, dtype: int64
Running on: cuda
Starting training...
Epoch 1/10:
Train Loss: 0.6014, Train Accuracy: 61.70%
Val Loss: 0.6540, Val Accuracy: 62.07%
Epoch 2/10:
Train Loss: 0.5227, Train Accuracy: 69.27%
Val Loss: 0.6476, Val Accuracy: 64.77%
Epoch 3/10:
Train Loss: 0.4933, Train Accuracy: 72.17%
Val Loss: 0.6786, Val Accuracy: 61.79%
Epoch 4/10:
Train Loss: 0.4500, Train Accuracy: 76.61%
Val Loss: 0.6513, Val Accuracy: 60.37%
Epoch 5/10:
Train Loss: 0.4028, Train Accuracy: 79.02%
Val Loss: 0.6831, Val Accuracy: 71.88%
Epoch 6/10:
Train Loss: 0.3529, Train Accuracy: 82.65%
Val Loss: 0.6479, Val Accuracy: 67.33%
Epoch 7/10:
Train Loss: 0.2958, Train Accuracy: 85.86%
Val Loss: 0.6540, Val Accuracy: 66.76%
Epoch 8/10:
Train Loss: 0.2620, Train Accuracy: 87.15%
Val Loss: 0.6862, Val Accuracy: 67.33%
Epoch 9/10:
Train Loss: 0.2491, Train Accuracy: 88.62%
Val Loss: 0.6971, Val Accuracy: 67.76%
Epoch 10/10:
Train Loss: 0.2497, Train Accuracy: 88.76%
Val Loss: 0.6983, Val Accuracy: 67.76%
Model saved

'''