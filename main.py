import torch
import torch.nn as nn
import torch.optim as optim
from src.config import Config
from src.dataset import load_data
from src.model import EmotionNetResNet50
from src.train_val import TrainValManager
from src.test import test_model
import os

def main():
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Load data
    train_loader, val_loader, test_loader = load_data(Config)
    
    # Initialize model
    model = EmotionNetResNet50(
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED,
        dropout_rate=Config.DROPOUT_RATE
    )
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Train and validate
    trainer = TrainValManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=Config)
    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.1, verbose=True
    )
    
    best_val_loss = trainer.train()
    
    # Test
    checkpoint = torch.load(Config.MODEL_SAVE_PATH)
    if 'model_state_dict' in checkpoint:
        # If the checkpoint contains a dictionary with model_state_dict
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If the checkpoint contains only the model state dict
        model.load_state_dict(checkpoint)
    test_model(model, test_loader, Config.DEVICE, Config.OUTPUT_DIR)

if __name__ == "__main__":
    main()
