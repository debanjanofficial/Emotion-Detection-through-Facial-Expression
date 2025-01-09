import torch
import torch.nn as nn
from tqdm import tqdm
import logging
from sklearn.metrics import f1_score
import os
from datetime import datetime
import numpy as np      


class TrainValManager:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = config.DEVICE
        self.num_epochs = config.NUM_EPOCHS
        self.patience = config.EARLY_STOPPING_PATIENCE
        self.config=config
        self.scheduler = None
        
        # Metrics history
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        
        # Early stopping variables
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0
        self.patience_counter = 0
        self.min_delta = 0.001
        self.train_val_diff_threshold = 0.1
        
        # Setup logging
        if not os.path.exists('logs'):
            os.makedirs('logs')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/training_{timestamp}.log'),
                logging.StreamHandler()
            ]
        )


    def train(self):
        logging.info("Starting Training...")
        for epoch in range(self.num_epochs):
            # Training phase
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc, val_f1 = self.validate_epoch()
            
            # Update learning rate using the validation loss value
            if self.scheduler:
                self.scheduler.step(val_loss)  # Pass the actual float valuex
                
            # Log metrics
            logging.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            logging.info(f"Training Loss: {train_loss:.4f}")
            logging.info(f"Training Accuracy: {train_acc:.2f}%")
            logging.info(f"Training F1 Score: {train_f1:.4f}")
            logging.info(f"Validation Loss: {val_loss:.4f}")
            logging.info(f"Validation Accuracy: {val_acc:.2f}%")
            logging.info(f"Validation F1 Score: {val_f1:.4f}")
            
            # Check early stopping with both training and validation metrics
            should_stop, is_improved = self.check_early_stopping(
                (train_loss, train_acc, train_f1),
                (val_loss, val_acc, val_f1)
            )
        
            # Save model if improved
            if is_improved:
                self.save_checkpoint(epoch, val_loss, val_acc, val_f1)
                logging.info("Saved model checkpoint")
            
            if should_stop:
                logging.info("Early stopping triggered")
                break
            
             
        return self.best_val_loss

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []
        
        train_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}/{self.num_epochs}")
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calculate metrics
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            running_loss += loss.item()
            
            # Update progress bar
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * np.mean(np.array(all_labels) == np.array(all_predictions))
        epoch_f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return epoch_loss, epoch_acc, epoch_f1


    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            val_bar = tqdm(self.val_loader, desc="Validating")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Get predictions
                _, predicted = torch.max(outputs.data, 1)
                
                # Store predictions and labels for metric calculation
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                running_loss += loss.item()  # Use item() to get the scalar value
                
                # Update progress bar
                val_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100 * np.mean(np.array(all_labels) == np.array(all_predictions))
        val_f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return val_loss, val_acc, val_f1  # Return actual values, not a dictionary

    def save_checkpoint(self, epoch, val_loss, val_acc, val_f1):
        """Save model checkpoint"""
        if not os.path.exists('models'):
            os.makedirs('models')
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'best_val_loss': self.best_val_loss
        }
    
        torch.save(checkpoint, "models/best_model.pth")

    def check_early_stopping(self, train_metrics, val_metrics):
        """
        Check if training should be stopped based on validation metrics
        Returns tuple of (should_stop, is_improved)
        """
        train_loss, train_acc, train_f1 = train_metrics
        val_loss, val_acc, val_f1 = val_metrics
        
        # Check if model is overfitting
        acc_diff = abs(train_acc - val_acc)
        f1_diff = abs(train_f1 - val_f1)
        
        # Check if validation loss improved
        is_improved = False
        if val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = val_loss
            self.patience_counter = 0
            is_improved = True
            logging.info(f"Validation loss improved to {val_loss:.4f}")
        else:
            self.patience_counter += 1
            logging.info(f"Validation loss did not improve. Counter: {self.patience_counter}/{self.patience}")
        
        # Check if we should stop
        should_stop = False
        if self.patience_counter >= self.patience:
            should_stop = True
            logging.info("Early stopping triggered")
        elif acc_diff > self.train_val_diff_threshold * 100 or f1_diff > self.train_val_diff_threshold:
            should_stop = True
            logging.info("Early stopping triggered due to overfitting")
        
        return should_stop, is_improved

