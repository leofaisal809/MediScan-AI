
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from pathlib import Path

device = torch.device("cpu")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "dataset" / "Chest_Xray_Bangladesh_Cleaned_v6_CANDIDATE" / "images"
MODEL_PATH = BASE_DIR / "chest_xray_model.pth"

# Prepare test images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_data = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=transform
)

test_loader = DataLoader(
    test_data,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

# Load trained model
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

class_names = checkpoint["class_names"]

if test_data.classes != class_names:
    raise ValueError("Test and training class names do not match!")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Evaluation
correct = 0
total = 0
confusion = [[0, 0], [0, 0]]

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        for actual, predicted in zip(labels, predictions):
            confusion[actual.item()][predicted.item()] += 1

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = (correct / total) * 100

print("\n===== TEST RESULTS =====")
print("Total test images:", total)
print(f"Test Accuracy: {accuracy:.2f}%")

print("\nConfusion Matrix:")
print("Rows = Actual | Columns = Predicted")
print("Class order:", class_names)
print(confusion)

for i, name in enumerate(class_names):
    true_positive = confusion[i][i]
    false_negative = sum(confusion[i]) - true_positive
    false_positive = sum(confusion[row][i] for row in range(2)) - true_positive

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0

    print(f"\n{name.upper()}")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")

print("\nEvaluation completed!")
