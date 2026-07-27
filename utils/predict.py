# ============================================
# predict.py
# Knee Arthritis Prediction
# ============================================

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ============================================
# Load Model
# ============================================

MODEL_PATH = "model/arthritis_model.h5"

model = load_model(MODEL_PATH)

# ============================================
# Class Labels
# ============================================

classes = [
    "Healthy",
    "Doubtful",
    "Mild",
    "Moderate",
    "Severe"
]

# ============================================
# Image Path
# ============================================

img_path = "test.jpg"

# ============================================
# Load and Preprocess Image
# ============================================

img = image.load_img(
    img_path,
    target_size=(160, 160)
)

img_array = image.img_to_array(img)

img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

# ============================================
# Prediction
# ============================================

prediction = model.predict(img_array)

predicted_index = np.argmax(prediction)

predicted_class = classes[predicted_index]

confidence = prediction[0][predicted_index] * 100

# ============================================
# Output
# ============================================

print("\n===================================")
print(" Knee Arthritis Prediction")
print("===================================")

print(f"\nPredicted Class : {predicted_class}")

print(f"Confidence      : {confidence:.2f}%")

print("\nClass Probabilities:")

for i, prob in enumerate(prediction[0]):
    print(f"{classes[i]} : {prob*100:.2f}%")
