import tensorflow as tf
import json
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from common import get_all_persons

os.makedirs("../models", exist_ok=True)

IMG = 224  # Same as face (MobileNetV2)
persons = get_all_persons()

if not persons:
    print("❌ No persons found in dataset!")
    exit(1)

print(f"🚀 Training Palm Recognition on {len(persons)} persons...")

gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10
)

train = gen.flow_from_directory(
    "../data/palm",  # CHANGED: palm directory
    target_size=(IMG, IMG),
    batch_size=16,
    subset="training",
    classes=persons,
    class_mode='categorical',
    shuffle=True
)

val = gen.flow_from_directory(
    "../data/palm",  # CHANGED: palm directory
    target_size=(IMG, IMG),
    batch_size=16,
    subset="validation",
    classes=persons,
    class_mode='categorical',
    shuffle=False
)

print(f"✅ Training samples: {train.samples}")
print(f"✅ Classes: {train.class_indices}")

# MobileNetV2 (same as face)
base = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG, IMG, 3)
)
base.trainable = False

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(persons), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("🏋️ Training model...")
history = model.fit(
    train,
    validation_data=val,
    epochs=10,
    verbose=1
)

# Save model and class mapping
model.save("../models/palm_model.h5")
with open("../models/palm_classes.json", "w") as f:
    json.dump(train.class_indices, f, indent=2)

print(f"✅ Palm model saved!")
print(f"✅ Final accuracy: {history.history['val_accuracy'][-1]:.2%}")