import torch
class Config:
    # Data parameters
    DATA_PATH = "data/fer2013.csv"
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Model parameters
    NUM_CLASSES = 7
    PRETRAINED = True
    DROPOUT_RATE = 0.3
    
    # Training parameters
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_PATIENCE = 5
    TRAIN_VAL_DIFF_THRESHOLD = 0.1  # Maximum allowed difference between train and validation metrics
    MIN_DELTA = 0.001  # Minimum improvement required
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Apple MPS or CPU fallback
    
    # Paths
    MODEL_SAVE_PATH = "models/best_model.pth"
    LOG_DIR = "logs"
    OUTPUT_DIR = "outputs"
