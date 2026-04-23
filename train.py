"""
Hyuk Jin Chung
4/22/2026

Training file for the MLP
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import GestureDataset
from model import HandGestureNet

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001

def main():
    """
    Train the model on custom hand gesture data
    - use the test data as validation set and manually adjust epochs for simplicity
    """
    # Load Data
    print("Loading datasets...")
    train_dataset = GestureDataset('gesture_dataset_split.csv', split_type='train')
    test_dataset = GestureDataset('gesture_dataset_split.csv', split_type='test')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandGestureNet().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Loss arrays for plotting
    train_losses = []
    val_losses = []

    print(f"\nStarting training on {device} for {EPOCHS} epochs...\n")

    # Training loop
    for epoch in range(EPOCHS):

        model.train()
        running_train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        correct_preds = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                
                # Calculate accuracy
                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels.data)
                
        epoch_val_loss = running_val_loss / len(test_dataset)
        val_losses.append(epoch_val_loss)
        
        val_acc = correct_preds.double() / len(test_dataset)

        # Console Output
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{EPOCHS}] | "
                  f"Train Loss: {epoch_train_loss:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f} | "
                  f"Val Accuracy: {val_acc*100:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), 'gesture_model.pth')
    print("\nTraining complete. Model saved to 'gesture_model.pth'")

    # Plot Results
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red', linestyle='--')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_plot.png')
    print("Loss plot saved to 'loss_plot.png'")
    plt.show()

if __name__ == "__main__":
    main()