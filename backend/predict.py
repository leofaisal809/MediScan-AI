import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path


# ==============================
# BASIC SETTINGS
# ==============================

device = torch.device("cpu")

BASE_DIR = Path(__file__).resolve().parent

XRAY_MODEL_PATH = BASE_DIR / "chest_xray_model.pth"
VALIDATOR_MODEL_PATH = BASE_DIR / "chest_xray_validator.pth"


# ==============================
# IMAGE TRANSFORMATION
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==============================
# CUSTOM ERROR
# ==============================

class InvalidImageError(Exception):
    pass


# ==============================
# LOAD CHEST X-RAY MODEL
# ==============================

xray_checkpoint = torch.load(
    XRAY_MODEL_PATH,
    map_location=device,
    weights_only=True
)

class_names = xray_checkpoint["class_names"]

xray_model = models.resnet18(weights=None)
xray_model.fc = nn.Linear(xray_model.fc.in_features, 2)

xray_model.load_state_dict(
    xray_checkpoint["model_state_dict"]
)

xray_model.to(device)
xray_model.eval()


# ==============================
# LOAD IMAGE VALIDATOR
# ==============================

validator_checkpoint = torch.load(
    VALIDATOR_MODEL_PATH,
    map_location=device,
    weights_only=True
)

validator_classes = validator_checkpoint["class_names"]

validator_model = models.resnet18(weights=None)
validator_model.fc = nn.Linear(
    validator_model.fc.in_features, 2
)

validator_model.load_state_dict(
    validator_checkpoint["model_state_dict"]
)

validator_model.to(device)
validator_model.eval()

if "chest_xray" not in validator_classes:
    raise RuntimeError(
        "Validator model does not contain the chest_xray class."
    )

if "non_chest" not in validator_classes:
    raise RuntimeError(
        "Validator model does not contain the non_chest class."
    )


# ==============================
# VALIDATE IMAGE TYPE
# ==============================

def validate_chest_xray(image_path):

    try:
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        image_tensor = transform(image).unsqueeze(0).to(device)

    except Exception as error:
        raise InvalidImageError(
            "Invalid image file. Please upload a valid chest X-ray."
        ) from error

    with torch.no_grad():

        outputs = validator_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, dim=1)

    predicted_class = validator_classes[predicted.item()]
    confidence = confidence.item() * 100

    if predicted_class != "chest_xray":

        raise InvalidImageError(
            "Invalid image. Please upload a chest X-ray."
        )

    return True, confidence


# ==============================
# PREDICT NORMAL OR PNEUMONIA
# ==============================

def predict_xray(image_path):

    # First check whether the uploaded image is a chest X-ray.
    validate_chest_xray(image_path)

    # Only then run the Normal/Pneumonia model.
    try:
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        image_tensor = transform(image).unsqueeze(0).to(device)

    except Exception as error:
        raise InvalidImageError(
            "Unable to read this image. Please upload a valid chest X-ray."
        ) from error

    with torch.no_grad():

        outputs = xray_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, dim=1)

    prediction = class_names[predicted.item()]
    confidence = confidence.item() * 100

    return prediction, confidence