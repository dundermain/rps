import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import yaml
import os
import numpy as np
from utils import save_model


with open("../config/config.yaml", "r") as file:
    config = yaml.safe_load(file)


def load_resnet_model(pretrained=True):                                 #loads a pretrained resnet-18 model 
    model = models.resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features                                     #num_ftrs indicates the number of input features to the final fully connected layer
    model.fc = nn.Linear(num_ftrs, 3)                                   #this is a new fc layer that takes num_ftrs and return 3 output features(rock, paper, scissor)
    return model


def split_data(data_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    total_size = len(dataset)
    
    train_size = int(config['data_params']['train_split'] * total_size)
    val_size = int(config['data_params']['val_split'] * total_size)
    test_size = total_size - train_size - val_size
    
    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def train_model_with_kfold(data_dir, k_folds, batch_size, lr, num_epochs, num_workers):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Cross-validation setup
    kfold = KFold(n_splits=k_folds, shuffle=True)
    results = {}

    # K-fold cross-validation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        # Sample elements randomly from a given list of indices
        train_subsampler = Subset(dataset, train_ids)
        val_subsampler = Subset(dataset, val_ids)

        # Define data loaders for training and validation
        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Initialize the model
        model = load_resnet_model(pretrained=config['model_params']['pretrained'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train and validate for each fold
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

        # Validation
        correct, total = 0, 0
        with torch.no_grad():
            model.eval()
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        print(f'Accuracy for fold {fold}: {accuracy}%')
        results[fold] = accuracy

    print('K-FOLD CROSS VALIDATION RESULTS')
    print('--------------------------------')
    sum_accuracy = 0.0
    for fold in results:
        print(f'Fold {fold}: {results[fold]} %')
        sum_accuracy += results[fold]

    print(f'Average: {sum_accuracy / len(results)} %')

    # Save the final trained model
    save_model(model)

if __name__ == "__main__":
    data_dir = '../data/'
    batch_size = config['train_params']['batch_size']
    lr = config['train_params']['learning_rate']
    num_epochs = config['train_params']['num_epochs']
    num_workers = config['train_params']['num_workers']
    k_folds = config['train_params']['k_folds']

    train_model_with_kfold(data_dir, k_folds, batch_size, lr, num_epochs, num_workers)
