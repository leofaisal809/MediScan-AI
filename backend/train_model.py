
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from pathlib import Path

# Use CPU
device = torch.device("cpu")

# Dataset location
DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "dataset"
    / "Chest_Xray_Bangladesh_Cleaned_v6_CANDIDATE"
    / "images"
)

# Image preparation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Load training images
train_data = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

# Separate validation data
val_data = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=test_transform
)

train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size

train_split, val_split = random_split(
    range(len(train_data)),
    [train_size, val_size]
)

train_dataset = Subset(train_data, train_split.indices)
val_dataset = Subset(val_data, val_split.indices)

train_loader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, num_workers=0
)

val_loader = DataLoader(
    val_dataset, batch_size=8, shuffle=False, num_workers=0
)

# Load pretrained ResNet18
print("Loading AI model...")

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze existing layers
for param in model.parameters():
    param.requires_grad = False

# Change final layer for our two classes
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Train the model
for epoch in range(3):

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

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Epoch {epoch + 1}/3")
    print(f"Training Loss: {total_loss / len(train_loader):.4f}")
    print(f"Validation Accuracy: {accuracy:.2f}%")

# Save trained model
MODEL_PATH = Path(__file__).resolve().parent / "chest_xray_model.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": train_data.classes
}, MODEL_PATH)

print("Training completed!")
print("Model saved at:", MODEL_PATH)
