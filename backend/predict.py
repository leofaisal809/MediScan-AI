
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import gc


# ==============================
# BASIC SETTINGS
# ==============================

device = torch.device("cpu")

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

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
# SINGLE MODEL INSTANCE
# ==============================

model = None
active_model = None

class_names = None
validator_classes = None


def load_model(model_type):

    global model
    global active_model
    global class_names
    global validator_classes

    if model_type == active_model:
        return

    if model is None:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.to(device)

    if model_type == "validator":
        model_path = VALIDATOR_MODEL_PATH
    else:
        model_path = XRAY_MODEL_PATH

    # Load checkpoint using memory mapping.
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
        mmap=True
    )

    if model_type == "validator":
        validator_classes = checkpoint["class_names"]
    else:
        class_names = checkpoint["class_names"]

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    active_model = model_type

    # Release checkpoint memory.
    del checkpoint
    gc.collect()


# ==============================
# IMAGE READING
# ==============================

def prepare_image(image_path):

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        image_tensor = transform(image).unsqueeze(0)

        return image_tensor.to(device)

    except Exception as error:
        raise InvalidImageError(
            "Invalid image. Please upload a valid chest X-ray."
        ) from error


# ==============================
# VALIDATE CHEST X-RAY
# ==============================

def validate_chest_xray(image_path):

    global validator_classes

    image_tensor = prepare_image(image_path)

    load_model("validator")

    if "chest_xray" not in validator_classes:
        raise RuntimeError(
            "Validator model does not contain chest_xray class."
        )

    if "non_chest" not in validator_classes:
        raise RuntimeError(
            "Validator model does not contain non_chest class."
        )

    with torch.inference_mode():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities, dim=1
        )

    predicted_class = validator_classes[predicted.item()]

    confidence = confidence.item() * 100

    if predicted_class != "chest_xray":

        raise InvalidImageError(
            "Invalid image. Please upload a chest X-ray."
        )

    return True, confidence


# ==============================
# NORMAL OR PNEUMONIA PREDICTION
# ==============================

def predict_xray(image_path):

    global class_names

    # First validate the uploaded image.
    validate_chest_xray(image_path)

    # Prepare the image for disease classification.
    image_tensor = prepare_image(image_path)

    # Switch the same model to disease classifier weights.
    load_model("xray")

    with torch.inference_mode():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities, dim=1
        )

    prediction = class_names[predicted.item()]

    confidence = confidence.item() * 100

    return prediction, confidence
