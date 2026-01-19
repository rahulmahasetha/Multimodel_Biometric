import os
import numpy as np
import json
from PIL import Image
import tensorflow as tf

print("="*70)
print("           COMPLETE MULTI-MODAL BIOMETRIC EVALUATION")
print("="*70)

# Load all models
print("\n📦 Loading models...")
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
    
    # Verify mappings
    if not (face_classes == finger_classes == palm_classes):
        print("⚠️  Class mappings differ but will proceed with individual testing")
    
    persons = list(face_classes.keys())
    print(f"✅ Loaded models for {len(persons)} persons")
    
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
        return 0.0
    except:
        return 0.0

# Evaluation parameters
THRESHOLD = 0.85

# Initialize metrics for each modality
metrics = {
    'face': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
    'finger': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
    'palm': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
    'fusion': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
}

print("\n" + "="*70)
print("GENUINE TESTS (Same Person - All Modalities)")
print("="*70)

# GENUINE TESTS
genuine_tests_conducted = 0

for person in persons:
    face_dir = f"../data/face/{person}"
    finger_dir = f"../data/finger/{person}"
    palm_dir = f"../data/palm/{person}"
    
    if not all(os.path.exists(d) for d in [face_dir, finger_dir, palm_dir]):
        continue
    
    face_imgs = [f for f in os.listdir(face_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    finger_imgs = [f for f in os.listdir(finger_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    palm_imgs = [f for f in os.listdir(palm_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not (face_imgs and finger_imgs and palm_imgs):
        continue
    
    # Test up to 3 samples per person
    n_tests = min(3, len(face_imgs), len(finger_imgs), len(palm_imgs))
    
    for i in range(n_tests):
        try:
            face_path = os.path.join(face_dir, face_imgs[i])
            finger_path = os.path.join(finger_dir, finger_imgs[i])
            palm_path = os.path.join(palm_dir, palm_imgs[i])
            
            # Get scores for THIS person
            face_score = get_person_score(face_model, face_classes, face_path, person, 224)
            finger_score = get_person_score(finger_model, finger_classes, finger_path, person, 128)
            palm_score = get_person_score(palm_model, palm_classes, palm_path, person, 224)
            
            # Fusion score
            fusion_score = 0.4*face_score + 0.3*finger_score + 0.3*palm_score
            
            # Update metrics for each modality
            for modality, score in [('face', face_score), 
                                   ('finger', finger_score), 
                                   ('palm', palm_score), 
                                   ('fusion', fusion_score)]:
                if score >= THRESHOLD:
                    metrics[modality]['TP'] += 1  # Correct accept
                else:
                    metrics[modality]['FN'] += 1  # Wrong reject
            
            genuine_tests_conducted += 1
            
        except Exception as e:
            print(f"  Skipping test {i} for {person}: {e}")

print(f"\n✅ Completed {genuine_tests_conducted} genuine tests")

print("\n" + "="*70)
print("IMPOSTER TESTS (Different Persons - All Modalities)")
print("="*70)

# IMPOSTER TESTS
imposter_tests_conducted = 0
max_imposter_tests = min(50, len(persons) * (len(persons)-1))

for i, p1 in enumerate(persons):
    for p2 in persons[i+1:]:
        if imposter_tests_conducted >= max_imposter_tests:
            break
            
        face_dir1 = f"../data/face/{p1}"
        finger_dir2 = f"../data/finger/{p2}"
        palm_dir2 = f"../data/palm/{p2}"
        
        if not all(os.path.exists(d) for d in [face_dir1, finger_dir2, palm_dir2]):
            continue
        
        try:
            # Get first image from each
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
            face_score = get_person_score(face_model, face_classes, face_path, p2, 224)
            # Person2's finger should NOT match Person1
            finger_score = get_person_score(finger_model, finger_classes, finger_path, p1, 128)
            # Person2's palm should NOT match Person1
            palm_score = get_person_score(palm_model, palm_classes, palm_path, p1, 224)
            
            # Fusion score
            fusion_score = 0.4*face_score + 0.3*finger_score + 0.3*palm_score
            
            # Update metrics for each modality
            for modality, score in [('face', face_score), 
                                   ('finger', finger_score), 
                                   ('palm', palm_score), 
                                   ('fusion', fusion_score)]:
                if score >= THRESHOLD:
                    metrics[modality]['FP'] += 1  # Wrong accept
                else:
                    metrics[modality]['TN'] += 1  # Correct reject
            
            imposter_tests_conducted += 1
            
        except Exception as e:
            print(f"  Skipping imposter test {p1} vs {p2}: {e}")

print(f"\n✅ Completed {imposter_tests_conducted} imposter tests")

# CALCULATE METRICS
print("\n" + "="*70)
print("                  COMPREHENSIVE RESULTS")
print("="*70)

results_table = []

for modality in ['face', 'finger', 'palm', 'fusion']:
    TP = metrics[modality]['TP']
    FP = metrics[modality]['FP']
    TN = metrics[modality]['TN']
    FN = metrics[modality]['FN']
    
    total_tests = TP + FP + TN + FN
    
    if total_tests == 0:
        continue
    
    # Calculate metrics
    accuracy = (TP + TN) / total_tests * 100
    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    far = FP / (FP + TN) * 100 if (FP + TN) > 0 else 0  # False Acceptance Rate
    frr = FN / (FN + TP) * 100 if (FN + TP) > 0 else 0  # False Rejection Rate
    
    # Store for table
    results_table.append({
        'modality': modality.upper() if modality != 'fusion' else 'HYBRID FUSION',
        'accuracy': accuracy,
        'far': far,
        'frr': frr,
        'precision': precision,
        'recall': recall,
        'tests': total_tests,
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN
    })

# Print detailed table
print("\n" + "-"*70)
print(f"{'MODALITY':<15} {'ACCURACY':<10} {'FAR':<10} {'FRR':<10} {'TESTS':<10}")
print("-"*70)

for result in results_table:
    print(f"{result['modality']:<15} {result['accuracy']:>7.2f}%  {result['far']:>7.2f}%  {result['frr']:>7.2f}%  {result['tests']:>8}")

print("-"*70)

# Print detailed metrics for each modality
print("\n" + "="*70)
print("                 DETAILED METRICS")
print("="*70)

for result in results_table:
    print(f"\n📊 {result['modality']}:")
    print(f"   Accuracy:  {result['accuracy']:.2f}%")
    print(f"   Precision: {result['precision']:.2f}%")
    print(f"   Recall:    {result['recall']:.2f}%")
    print(f"   FAR:       {result['far']:.2f}%")
    print(f"   FRR:       {result['frr']:.2f}%")
    print(f"   Tests:     {result['tests']} (TP:{result['TP']}, FP:{result['FP']}, TN:{result['TN']}, FN:{result['FN']})")

# Performance analysis
print("\n" + "="*70)
print("                 PERFORMANCE ANALYSIS")
print("="*70)

# Find best performing modality
if results_table:
    best_accuracy = max(results_table, key=lambda x: x['accuracy'])
    best_security = min(results_table, key=lambda x: x['far'])
    best_usability = min(results_table, key=lambda x: x['frr'])
    
    print(f"\n🏆 BEST PERFORMANCE:")
    print(f"   Highest Accuracy: {best_accuracy['modality']} ({best_accuracy['accuracy']:.2f}%)")
    print(f"   Best Security (Lowest FAR): {best_security['modality']} ({best_security['far']:.2f}%)")
    print(f"   Best Usability (Lowest FRR): {best_usability['modality']} ({best_usability['frr']:.2f}%)")
    
    # Fusion improvement
    fusion_result = next((r for r in results_table if r['modality'] == 'HYBRID FUSION'), None)
    if fusion_result and len(results_table) > 1:
        avg_individual_accuracy = np.mean([r['accuracy'] for r in results_table if r['modality'] != 'HYBRID FUSION'])
        improvement = fusion_result['accuracy'] - avg_individual_accuracy
        
        print(f"\n📈 FUSION IMPROVEMENT:")
        print(f"   Average Individual Modality Accuracy: {avg_individual_accuracy:.2f}%")
        print(f"   Hybrid Fusion Accuracy: {fusion_result['accuracy']:.2f}%")
        print(f"   Improvement: {improvement:+.2f}%")
        
        if improvement > 0:
            print(f"   ✅ Fusion provides {improvement:.2f}% accuracy gain over individual modalities")
        else:
            print(f"   ⚠️  Fusion shows no improvement over individual modalities")

# Security vs Usability Trade-off
print("\n" + "="*70)
print("           SECURITY vs USABILITY TRADE-OFF")
print("="*70)

if results_table:
    fusion_result = next((r for r in results_table if r['modality'] == 'HYBRID FUSION'), None)
    if fusion_result:
        if fusion_result['far'] < 1 and fusion_result['frr'] < 5:
            print(f"✅ EXCELLENT: High security (FAR={fusion_result['far']:.2f}%) with good usability (FRR={fusion_result['frr']:.2f}%)")
        elif fusion_result['far'] < 5 and fusion_result['frr'] < 10:
            print(f"⚠️  GOOD: Moderate security (FAR={fusion_result['far']:.2f}%) with acceptable usability (FRR={fusion_result['frr']:.2f}%)")
        elif fusion_result['far'] < 10:
            print(f"⚠️  FAIR: Low security (FAR={fusion_result['far']:.2f}%) - consider lowering threshold")
        else:
            print(f"❌ POOR: Security compromised (FAR={fusion_result['far']:.2f}%) - system needs improvement")

print("\n" + "="*70)
print("                 EVALUATION COMPLETE")
print("="*70)
print(f"\n📋 Summary:")
print(f"   Total Tests: {genuine_tests_conducted + imposter_tests_conducted}")
print(f"   Genuine Tests: {genuine_tests_conducted}")
print(f"   Imposter Tests: {imposter_tests_conducted}")
print(f"   Threshold Used: {THRESHOLD}")
print(f"   Persons in Dataset: {len(persons)}")
print("="*70)