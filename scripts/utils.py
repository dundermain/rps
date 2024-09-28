import torch
import cv2
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim

def load_resnet_model():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    return model

def load_trained_model(model_path="../models/model.pth"):
    model = load_resnet_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def save_model(model, path="../models/model.pth"):
    torch.save(model.state_dict(), path)

def detect_rps_sign(model, frame):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = transform(frame_rgb).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return ["rock", "paper", "scissor"][predicted.item()]

def get_computer_choice():
    import random
    return random.choice(["rock", "paper", "scissor"])

def determine_winner(player_choice, computer_choice):
    if player_choice == computer_choice:
        return "Tie"
    elif (player_choice == "rock" and computer_choice == "scissor") or \
         (player_choice == "scissor" and computer_choice == "paper") or \
         (player_choice == "paper" and computer_choice == "rock"):
        return "Player"
    else:
        return "Computer"
    
def save_model(model, path="../models/model.pth"):
    torch.save(model.state_dict(), path)

def retrain_model_with_frame(model, frame, label):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = transform(frame_rgb).unsqueeze(0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    output = model(image)
    target = torch.tensor([["rock", "paper", "scissor"].index(label)])  
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    save_model(model)
    print("Model retrained with new frame.")
