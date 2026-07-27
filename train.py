# ============================================================
# Hybrid VGG19 + Vision Transformer
# Knee Arthritis Detection
# ============================================================

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
    Concatenate,
    GlobalAveragePooling2D,
    LayerNormalization,
    MultiHeadAttention,
    Add
)

from tensorflow.keras.models import Model

from tensorflow.keras.applications import VGG19

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ============================================================
# PARAMETERS
# ============================================================

DATASET_PATH = "dataset"

IMG_SIZE = 160

BATCH_SIZE = 8

EPOCHS = 5

NUM_CLASSES = 5

# ============================================================
# DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# ============================================================
# TRAIN GENERATOR
# ============================================================

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# ============================================================
# VALIDATION GENERATOR
# ============================================================

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights = dict(enumerate(class_weights))

print("\nClass Weights:")
print(class_weights)

# ============================================================
# INPUT LAYER
# ============================================================

inputs = Input(shape=(160,160,3))

# ============================================================
# VGG19 FEATURE EXTRACTION
# ============================================================

vgg_base = VGG19(
    weights='imagenet',
    include_top=False,
    input_tensor=inputs
)

# Freeze layers
for layer in vgg_base.layers:
    layer.trainable = False

vgg_features = vgg_base.output

vgg_features = GlobalAveragePooling2D()(vgg_features)

# ============================================================
# SIMPLE VISION TRANSFORMER BLOCK
# ============================================================

x = Flatten()(inputs)

x = Dense(256)(x)

x = tf.expand_dims(x, axis=1)

attention_output = MultiHeadAttention(
    num_heads=4,
    key_dim=64
)(x, x)

x = Add()([x, attention_output])

x = LayerNormalization()(x)

x = Flatten()(x)

x = Dense(256, activation='relu')(x)

# ============================================================
# COMBINE VGG19 + ViT FEATURES
# ============================================================

combined = Concatenate()([
    vgg_features,
    x
])

# ============================================================
# CLASSIFICATION HEAD
# ============================================================

x = Dense(512, activation='relu')(combined)

x = Dropout(0.5)(x)

x = Dense(256, activation='relu')(x)

x = Dropout(0.3)(x)

outputs = Dense(NUM_CLASSES, activation='softmax')(x)

# ============================================================
# FINAL MODEL
# ============================================================

model = Model(inputs=inputs, outputs=outputs)

# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-7
)

checkpoint = ModelCheckpoint(
    "best_hybrid_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

callbacks = [
    early_stop,
    reduce_lr,
    checkpoint
]

# ============================================================
# TRAIN MODEL
# ============================================================

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs("model", exist_ok=True)

model.save("model/hybrid_vgg19_vit_model.h5")

print("\nModel Saved Successfully!")

# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10,5))

plt.plot(history.history['accuracy'], label='Training Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.savefig("accuracy_graph.png")

plt.show()

# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(10,5))

plt.plot(history.history['loss'], label='Training Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.savefig("loss_graph.png")

plt.show()

# ============================================================
# EVALUATE MODEL
# ============================================================

loss, accuracy = model.evaluate(validation_generator)

print(f"\nValidation Accuracy: {accuracy*100:.2f}%")

# ============================================================
# PREDICTIONS
# ============================================================

validation_generator.reset()

Y_pred = model.predict(validation_generator)

y_pred = np.argmax(Y_pred, axis=1)

y_true = validation_generator.classes

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average='weighted'
)

recall = recall_score(
    y_true,
    y_pred,
    average='weighted'
)

f1 = f1_score(
    y_true,
    y_pred,
    average='weighted'
)

print("\n================================")

print(f"Accuracy  : {accuracy*100:.2f}%")

print(f"Precision : {precision*100:.2f}%")

print(f"Recall    : {recall*100:.2f}%")

print(f"F1 Score  : {f1*100:.2f}%")

print("================================")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

class_names = list(validation_generator.class_indices.keys())

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")

plt.show()

print("\nTraining Completed Successfully!")