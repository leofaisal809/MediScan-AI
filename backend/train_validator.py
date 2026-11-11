
import random
from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image, UnidentifiedImageError

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


# ==========================================
# 1. SETTINGS
# ==========================================

random.seed(42)
torch.manual_seed(42)

device = torch.device("cpu")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

XRAY_DIR = PROJECT_DIR / "dataset" / "chest-xray" / "train"
PHOTO_DIR = PROJECT_DIR / "dataset" / "non chest x-ray"

OLD_MODEL = BASE_DIR / "chest_xray_model.pth"
SAVE_PATH = BASE_DIR / "chest_xray_validator.pth"

EPOCHS = 3
BATCH_SIZE = 16

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".tif", ".tiff"
}


# ==========================================
# 2. CHECK FOLDERS
# ==========================================

print("\n========== FOLDER CHECK ==========")

print("Project:", PROJECT_DIR)
print("X-ray folder:", XRAY_DIR)
print("Photo folder:", PHOTO_DIR)

if not XRAY_DIR.exists():
    raise FileNotFoundError("Chest X-ray folder not found!")

if not PHOTO_DIR.exists():
    raise FileNotFoundError("Ordinary photo folder not found!")

if not OLD_MODEL.exists():
    raise FileNotFoundError("Original chest X-ray model not found!")

print("All folders found successfully!")


# ==========================================
# 3. IMAGE TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==========================================
# 4. LOAD CHEST X-RAYS
# ==========================================

print("\n========== LOADING CHEST X-RAYS ==========")

xray_dataset = ImageFolder(
    root=str(XRAY_DIR)
)

xray_paths = [
    Path(path)
    for path, label in xray_dataset.samples
]

print("X-ray categories:", xray_dataset.classes)
print("X-ray images found:", len(xray_paths))


# ==========================================
# 5. LOAD ORDINARY PHOTOGRAPHS
# ==========================================

print("\n========== LOADING ORDINARY PHOTOS ==========")

# Search all nested folders recursively.
# This handles the extra Caltech-101 folder.

photo_paths = [
    path
    for path in PHOTO_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
]

print("Ordinary photos found:", len(photo_paths))


# ==========================================
# 6. REMOVE INVALID IMAGE FILES
# ==========================================

print("\n========== CHECKING IMAGE FILES ==========")

def valid_image(path):

    try:
        with Image.open(path) as image:
            image.verify()

        return True

    except Exception:
        return False


xray_paths = [
    path for path in xray_paths
    if valid_image(path)
]

photo_paths = [
    path for path in photo_paths
    if valid_image(path)
]

print("Valid X-rays:", len(xray_paths))
print("Valid ordinary photos:", len(photo_paths))

if len(xray_paths) < 10 or len(photo_paths) < 10:
    raise ValueError(
        "Not enough valid images in one or both datasets."
    )


# ==========================================
# 7. BALANCE DATASET
# ==========================================

random.shuffle(xray_paths)
random.shuffle(photo_paths)

sample_count = min(
    len(xray_paths),
    len(photo_paths)
)

xray_paths = xray_paths[:sample_count]
photo_paths = photo_paths[:sample_count]

print("\n========== BALANCED DATASET ==========")

print("Chest X-rays:", len(xray_paths))
print("Ordinary photos:", len(photo_paths))


# ==========================================
# 8. CUSTOM DATASET
# ==========================================

class ValidatorDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("RGB")

            image = transform(image)

        return image, label


# Label 0 = Chest X-ray
# Label 1 = Ordinary photograph

all_samples = []

for path in xray_paths:
    all_samples.append((path, 0))

for path in photo_paths:
    all_samples.append((path, 1))

random.shuffle(all_samples)

dataset = ValidatorDataset(all_samples)


# ==========================================
# 9. TRAIN / VALIDATION SPLIT
# ==========================================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_data, val_data = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nTraining images:", train_size)
print("Validation images:", val_size)


# ==========================================
# 10. BUILD VALIDATOR MODEL
# ==========================================

print("\n========== BUILDING VALIDATOR ==========")

checkpoint = torch.load(
    OLD_MODEL,
    map_location=device,
    weights_only=True
)

old_weights = checkpoint["model_state_dict"]

model = models.resnet18(weights=None)

# Reuse existing chest X-ray model features.
# The original model file is not modified.

feature_weights = {
    key: value
    for key, value in old_weights.items()
    if not key.startswith("fc.")
}

model.load_state_dict(
    feature_weights,
    strict=False
)

# Two output classes:
# 0 = chest_xray
# 1 = non_chest

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

# Freeze earlier layers.

for parameter in model.parameters():
    parameter.requires_grad = False

# Train the last ResNet block.

for parameter in model.layer4.parameters():
    parameter.requires_grad = True

# Train the new classification layer.

for parameter in model.fc.parameters():
    parameter.requires_grad = True

model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=0.0001
)


# ==========================================
# 11. TRAIN MODEL
# ==========================================

print("\n========== STARTING TRAINING ==========")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)


    # ======================================
    # VALIDATION
    # ======================================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {average_loss:.4f} | "
        f"Validation Accuracy: {accuracy:.2f}%"
    )


# ==========================================
# 12. SAVE VALIDATOR
# ==========================================

print("\n========== SAVING VALIDATOR ==========")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": [
            "chest_xray",
            "non_chest"
        ]
    },
    SAVE_PATH
)

print("\n====================================")
print("VALIDATOR TRAINING COMPLETED!")
print("====================================")

print("Saved model:", SAVE_PATH)

print("\nOriginal Normal/Pneumonia model remains unchanged.")

