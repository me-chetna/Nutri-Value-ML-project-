"""CNN regressor for nutrition-label images.

This network is NOT useful until trained with paired images and nutrition values.
See train_cnn.py and README.md. It predicts seven numeric values; it does not
read arbitrary printed text as reliably as OCR without a suitable dataset.
"""
import torch
from torch import nn


class FoodNutritionCNN(nn.Module):
    def __init__(self, output_features=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(192 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.25), nn.Linear(256, output_features)
        )

    def forward(self, images):
        return self.regressor(self.features(images))
