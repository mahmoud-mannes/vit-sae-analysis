import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from main.prep_data import prep_data
from main.model import predict

def evaluate_robustness(model, processor, normal_dataset, source, RPI = False, magnitude = 1.0, corruption_type = "Gaussian Blur"):
    # Prepare dataloaders for the JPEG-corrupted and normal datasets
    
    dataloader_shifted = prep_data(normal_dataset,processor, source, corruption_type = corruption_type, severity = 5)
    dataloader_normal = prep_data(normal_dataset,processor, source)

    # Evaluate accuracy on both procured datasets

    mean_acc = predict(model, dataloader_normal, source, RPI, magnitude)
    shifted_acc = predict(model, dataloader_shifted, source, RPI, magnitude)

    return (1 - (shifted_acc/mean_acc))  # Evaluate and return fragility score
