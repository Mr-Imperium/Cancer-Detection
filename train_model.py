import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.optim import Adam
from PIL import Image
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class CancerDataset(Dataset):
    def __init__(self, data_dir, labels_path, transform, dataset_type=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Get list of image files
        self.filenames = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
        
        # Load labels
        self.labels_df = pd.read_csv(labels_path)
        self.labels_df.set_index("id", inplace=True)
        
        # Split dataset based on type
        if dataset_type == "train":
            self.filenames = self.filenames[:2608]
        elif dataset_type == "val":
            self.filenames = self.filenames[2608:2708]
        elif dataset_type == "test":
            self.filenames = self.filenames[2708:]
            
        self.labels = [self.labels_df.loc[filename[:-4]].values[0] for filename in self.filenames]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.filenames[idx])
        img = Image.open(img_path)
        img = self.transform(img)
        return img, self.labels[idx]

def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    
    return model

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
    return accuracy

def main():
    # Paths
    data_dir = "data/data_sample"
    labels_path = "data/labels.csv"
    
    # Data transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Create datasets
    train_dataset = CancerDataset(data_dir, labels_path, train_transform, "train")
    test_dataset = CancerDataset(data_dir, labels_path, test_transform, "test")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Initialize model
    model = models.resnet34(pretrained=True)
    
    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=3e-4)
    
    # Train model
    model = train_model(model, train_loader, criterion, optimizer)
    
    # Evaluate model
    accuracy = evaluate_model(model, test_loader)
    
    # Save model
    torch.save(model.state_dict(), 'cancer_detection_model.pth')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()