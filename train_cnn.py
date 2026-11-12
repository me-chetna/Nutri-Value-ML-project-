"""Train CNN regression model from a labelled image CSV.

CSV columns: image_path, calories, sugar, protein, fat, sodium, carbohydrates, fiber.
All nutrition values must be per 100 g/ml. Image paths are relative to this project
folder or absolute paths. At least several hundred diverse labelled images are
recommended; more is better.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

from cnn_model import FoodNutritionCNN

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "cnn_dataset.csv"
FEATURES = ["calories", "sugar", "protein", "fat", "sodium", "carbohydrates", "fiber"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NutritionImageDataset(Dataset):
    def __init__(self, frame, scaler):
        self.frame = frame.reset_index(drop=True)
        self.scaler = scaler
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = ROOT / path
        with Image.open(path) as im:
            image = self.transform(im.convert("RGB"))
        target = self.scaler.transform([[float(row[f]) for f in FEATURES]])[0]
        return image, torch.tensor(target, dtype=torch.float32)


def main():
    if not CSV.exists():
        raise FileNotFoundError(f"Create {CSV.name}; see README.md for the required columns.")
    frame = pd.read_csv(CSV)
    required = ["image_path", *FEATURES]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError("Missing columns in cnn_dataset.csv: " + ", ".join(missing))
    frame = frame.dropna(subset=required).copy()
    if len(frame) < 20:
        raise ValueError("At least 20 labelled images are required to start training; use substantially more for meaningful results.")
    for f in FEATURES:
        frame[f] = pd.to_numeric(frame[f], errors="coerce")
    frame = frame.dropna(subset=FEATURES)
    scaler = StandardScaler().fit(frame[FEATURES].values)
    dataset = NutritionImageDataset(frame, scaler)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
    model = FoodNutritionCNN(len(FEATURES)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.SmoothL1Loss()
    best_val = float("inf")
    for epoch in range(25):
        model.train(); train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad(); loss = loss_fn(model(images), targets); loss.backward(); optimizer.step()
            train_loss += loss.item() * len(images)
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                val_loss += loss_fn(model(images), targets).item() * len(images)
        val_loss /= len(val_set)
        print(f"Epoch {epoch+1:02d}/25 | train={train_loss/train_size:.4f} | val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.cpu().state_dict(), ROOT / "models" / "cnn_nutrition_model.pth")
            model.to(DEVICE)
            joblib.dump(scaler, ROOT / "models" / "cnn_target_scaler.pkl")
    print("Best validation loss:", best_val)
    print("CNN saved in models/. Evaluate on a separate real-world test set before relying on predictions.")

if __name__ == "__main__":
    (ROOT / "models").mkdir(exist_ok=True)
    main()
