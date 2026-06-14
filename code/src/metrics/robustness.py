import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from main.prep_data import prep_data
from main.model import predict

def evaluate_robustness_jpeg(model, normal_dataset, shifted_dataset, RPI = False, magnitude = 1.0):
    # Prepare dataloaders for the JPEG-corrupted and normal datasets
    
    dataloader_shifted = prep_data(shifted_dataset)
    dataloader_normal = prep_data(normal_dataset)

    # Evaluate accuracy on both procured datasets

    mean_acc = predict(model, dataloader_normal, RPI, magnitude)
    shifted_acc = predict(model, dataloader_shifted, RPI, magnitude)

    return (1 - (shifted_acc/mean_acc))  # Evaluate and return fragility score
