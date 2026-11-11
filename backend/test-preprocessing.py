from preprocessing import preprocess_image

image_path = "backend/uploads/test.png"

processed_image = preprocess_image(image_path)

print("Image processed successfully!")
print("Shape:", processed_image.shape)
print("Minimum pixel value:", processed_image.min())
print("Maximum pixel value:", processed_image.max())