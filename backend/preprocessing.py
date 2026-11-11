from PIL import Image
import numpy as np


def preprocess_image(image_path):
    # Open the X-ray image
    image = Image.open(image_path)

    # Convert image to grayscale
    image = image.convert("L")

    # Resize image to 224 x 224
    image = image.resize((224, 224))

    # Convert image to NumPy array
    image_array = np.array(image, dtype=np.float32)

    # Normalize pixel values from 0-255 to 0-1
    image_array = image_array / 255.0

    return image_array