import tensorflow as tf
import json
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from common import get_all_persons

os.makedirs("../models", exist_ok=True)

IMG = 128
persons = get_all_persons()

if not persons:
    print("❌ No persons found in dataset!")
    exit(1)

print(f"🚀 Training Fingerprint Recognition on {len(persons)} persons...")

gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train = gen.flow_from_directory(
    "../data/finger",
    target_size=(IMG, IMG),
    batch_size=16,
    subset="training",
    classes=persons,
    class_mode='categorical',  # FIXED
    shuffle=True
)

val = gen.flow_from_directory(
    "../data/finger",
    target_size=(IMG, IMG),
    batch_size=16,
    subset="validation",
    classes=persons,
    class_mode='categorical',  # FIXED
    shuffle=False
)

print(f"✅ Training samples: {train.samples}")
print(f"✅ Classes: {train.class_indices}")

# Custom CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMG, IMG, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(persons), activation="softmax")
])

model.compile(
    optimizer="adam",
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
model.save("../models/finger_model.h5")
with open("../models/finger_classes.json", "w") as f:
    json.dump(train.class_indices, f, indent=2)

print(f"✅ Fingerprint model saved!")
print(f"✅ Final accuracy: {history.history['val_accuracy'][-1]:.2%}")