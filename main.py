from src.data_preprocessing import preprocess_data
from src.train import train_models

print("Step 1: Data Preprocessing")
preprocess_data()

print("Step 2: Training Models")
train_models()

print("Training Completed")