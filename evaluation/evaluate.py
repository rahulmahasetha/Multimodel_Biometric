import os
import numpy as np
import json
from PIL import Image
import tensorflow as tf

# Load models
print("📊 Loading models for evaluation...")
try:
    face_model = tf.keras.models.load_model("../models/face_model.h5")
    finger_model = tf.keras.models.load_model("../models/finger_model.h5")
    palm_model = tf.keras.models.load_model("../models/palm_model.h5")
    
    with open("../models/face_classes.json", "r") as f:
        face_classes = json.load(f)
    with open("../models/finger_classes.json", "r") as f:
        finger_classes = json.load(f)
    with open("../models/palm_classes.json", "r") as f:
        palm_classes = json.load(f)
    
    # Verify all mappings are identical
    if not (face_classes == finger_classes == palm_classes):
        print("❌ WARNING: Class mappings are not identical across modalities!")
        print("   This will affect fusion accuracy!")
    
    persons = list(face_classes.keys())
    print(f"✅ Loaded models for {len(persons)} persons: {persons}")
    
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    exit(1)

def get_person_score(model, class_mapping, image_path, person_id, img_size):
    """Get probability that image belongs to specific person"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((img_size, img_size))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        
        pred = model.predict(arr, verbose=0)[0]
        
        if person_id in class_mapping:
            person_idx = class_mapping[person_id]
            return float(pred[person_idx])
        else:
            return 0.0
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0.0

# Evaluation parameters
THRESHOLD = 0.85
metrics = {
    'genuine_tests': 0,
    'genuine_accept': 0,
    'imposter_tests': 0,
    'imposter_accept': 0
}

print("\n" + "="*60)
print("GENUINE TESTS (Same Person)")
print("="*60)

# Genuine tests: same person for all modalities
for person in persons:
    face_dir = f"../data/face/{person}"
    finger_dir = f"../data/finger/{person}"
    palm_dir = f"../data/palm/{person}"
    
    # Skip if any modality missing
    if not all(os.path.exists(d) for d in [face_dir, finger_dir, palm_dir]):
        print(f"⚠️  Skipping {person}: missing modality data")
        continue
    
    # Get all images
    face_images = [f for f in os.listdir(face_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    finger_images = [f for f in os.listdir(finger_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    palm_images = [f for f in os.listdir(palm_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not (face_images and finger_images and palm_images):
        print(f"⚠️  Skipping {person}: no images found")
        continue
    
    # Test up to 3 samples per person (or min available)
    n_samples = min(3, len(face_images), len(finger_images), len(palm_images))
    
    for i in range(n_samples):
        try:
            face_path = os.path.join(face_dir, face_images[i])
            finger_path = os.path.join(finger_dir, finger_images[i])
            palm_path = os.path.join(palm_dir, palm_images[i])
            
            # Get scores for this person
            face_score = get_person_score(face_model, face_classes, face_path, person, 224)
            finger_score = get_person_score(finger_model, finger_classes, finger_path, person, 128)
            palm_score = get_person_score(palm_model, palm_classes, palm_path, person, 224)
            
            # Fusion
            fusion_score = 0.4*face_score + 0.3*finger_score + 0.3*palm_score
            
            metrics['genuine_tests'] += 1
            if fusion_score >= THRESHOLD:
                metrics['genuine_accept'] += 1
                
        except Exception as e:
            print(f"❌ Error testing {person} sample {i}: {e}")

print("\n" + "="*60)
print("IMPOSTER TESTS (Different Persons)")
print("="*60)

# Imposter tests: different persons
imposter_tests_done = 0

for i, person1 in enumerate(persons):
    for person2 in persons[i+1:]:  # Avoid duplicate tests
        if imposter_tests_done >= 50:  # Limit to 50 tests for speed
            break
            
        face_dir1 = f"../data/face/{person1}"
        finger_dir2 = f"../data/finger/{person2}"
        palm_dir2 = f"../data/palm/{person2}"
        
        if not all(os.path.exists(d) for d in [face_dir1, finger_dir2, palm_dir2]):
            continue
        
        # Get first image from each
        try:
            face_img = [f for f in os.listdir(face_dir1) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))][0]
            finger_img = [f for f in os.listdir(finger_dir2) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))][0]
            palm_img = [f for f in os.listdir(palm_dir2) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))][0]
            
            face_path = os.path.join(face_dir1, face_img)
            finger_path = os.path.join(finger_dir2, finger_img)
            palm_path = os.path.join(palm_dir2, palm_img)
            
            # Person1's face should NOT match Person2
            face_score = get_person_score(face_model, face_classes, face_path, person2, 224)
            # Person2's finger should NOT match Person1
            finger_score = get_person_score(finger_model, finger_classes, finger_path, person1, 128)
            # Person2's palm should NOT match Person1
            palm_score = get_person_score(palm_model, palm_classes, palm_path, person1, 224)
            
            # Fusion
            fusion_score = 0.4*face_score + 0.3*finger_score + 0.3*palm_score
            
            metrics['imposter_tests'] += 1
            imposter_tests_done += 1
            
            if fusion_score >= THRESHOLD:
                metrics['imposter_accept'] += 1
                
        except Exception as e:
            print(f"❌ Error in imposter test {person1} vs {person2}: {e}")

# Calculate metrics
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

if metrics['genuine_tests'] > 0:
    accuracy = (metrics['genuine_accept'] / metrics['genuine_tests']) * 100
    frr = 100 - accuracy
else:
    accuracy = frr = 0

if metrics['imposter_tests'] > 0:
    far = (metrics['imposter_accept'] / metrics['imposter_tests']) * 100
else:
    far = 0

print(f"\n📈 HYBRID FUSION SYSTEM:")
print(f"   Accuracy:  {accuracy:.2f}%")
print(f"   FAR:       {far:.2f}%")
print(f"   FRR:       {frr:.2f}%")
print(f"   Threshold: {THRESHOLD}")
print(f"\n📊 Test Statistics:")
print(f"   Genuine tests: {metrics['genuine_tests']} (Accepted: {metrics['genuine_accept']})")
print(f"   Imposter tests: {metrics['imposter_tests']} (Accepted: {metrics['imposter_accept']})")

# Security vs Usability analysis
print(f"\n🔒 SECURITY ANALYSIS:")
if far < 1:
    print("   ✅ Excellent security (FAR < 1%)")
elif far < 5:
    print("   ⚠️  Good security (FAR < 5%)")
elif far < 10:
    print("   ⚠️  Moderate security (FAR < 10%)")
else:
    print("   ❌ Poor security (FAR >= 10%)")

print(f"\n👥 USABILITY ANALYSIS:")
if frr < 5:
    print("   ✅ Excellent usability (FRR < 5%)")
elif frr < 10:
    print("   ⚠️  Good usability (FRR < 10%)")
elif frr < 20:
    print("   ⚠️  Moderate usability (FRR < 20%)")
else:
    print("   ❌ Poor usability (FRR >= 20%)")

print("="*60)