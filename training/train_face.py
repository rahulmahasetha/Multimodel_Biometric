import tensorflow as tf
import json
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from common import get_all_persons

# =============================
# BASIC SETUP
# =============================
IMG = 224
BATCH_SIZE = 16
EPOCHS = 30
DATA_PATH = "../data/augmented_face"
MODEL_PATH = "../models"

os.makedirs(MODEL_PATH, exist_ok=True)

persons = get_all_persons()
if not persons:
    raise RuntimeError("❌ No persons found in dataset!")

print(f"🚀 Training Face Recognition on {len(persons)} persons")
print("🧑 Classes:", persons)

# =============================
# DATA GENERATOR
# =============================
gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train = gen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG, IMG),
    batch_size=BATCH_SIZE,
    subset="training",
    classes=persons,
    class_mode="categorical",
    shuffle=True
)

val = gen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG, IMG),
    batch_size=BATCH_SIZE,
    subset="validation",
    classes=persons,
    class_mode="categorical",
    shuffle=False
)

print(f"✅ Train samples: {train.samples}")
print(f"✅ Validation samples: {val.samples}")
print(f"✅ Class mapping: {train.class_indices}")

# =============================
# HANDLE EMPTY VALIDATION SAFELY
# =============================
USE_VALIDATION = val.samples > 0

if not USE_VALIDATION:
    print("⚠️ Validation set is EMPTY → training without validation")

# =============================
# BASE MODEL
# =============================
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG, IMG, 3)
)
base_model.trainable = False

# =============================
# FULL MODEL
# =============================
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(persons), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =============================
# CALLBACKS
# =============================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="loss" if not USE_VALIDATION else "val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss" if not USE_VALIDATION else "val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-6
    )
]

# =============================
# TRAINING PHASE 1
# =============================
print("🏋️ Training model (feature extraction)...")

history = model.fit(
    train,
    validation_data=val if USE_VALIDATION else None,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# =============================
# FINE-TUNING PHASE
# =============================
print("🔥 Fine-tuning MobileNetV2...")

base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train,
    validation_data=val if USE_VALIDATION else None,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

# =============================
# SAVE MODEL
# =============================
model.save(os.path.join(MODEL_PATH, "face_model.h5"))

with open(os.path.join(MODEL_PATH, "face_classes.json"), "w") as f:
    json.dump(train.class_indices, f, indent=2)

print("✅ Face model saved successfully!")

if USE_VALIDATION:
    print(f"🎯 Final validation accuracy: {history_fine.history['val_accuracy'][-1]:.2%}")
else:
    print(f"🎯 Final training accuracy: {history_fine.history['accuracy'][-1]:.2%}")
