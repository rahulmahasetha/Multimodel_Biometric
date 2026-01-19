import os
import sys
import cv2
from tkinter import Tk, filedialog
from inference.verify_face import verify as verify_face
from inference.verify_finger import verify as verify_finger
from inference.verify_palm import verify as verify_palm
from fusion.weighted_fusion import weighted_fusion as fuse

def check_models_exist():
    """Check if trained models exist"""
    required_models = [
        "models/face_model.h5",
        "models/finger_model.h5", 
        "models/palm_model.h5",
        "models/face_classes.json"
    ]
    
    missing_models = [m for m in required_models if not os.path.exists(m)]
    
    if missing_models:
        print("❌ Missing trained models:")
        for m in missing_models:
            print(f"   - {m}")
        print("\nPlease train models first:")
        print("cd training && python train_face.py")
        print("cd training && python train_finger.py")
        print("cd training && python train_palm.py")
        return False
    return True

def create_input_folder():
    """Create input folder if it doesn't exist"""
    os.makedirs("input", exist_ok=True)

def capture_from_camera(modality):
    """
    Capture image from webcam for a specific modality
    
    Args:
        modality: "face", "finger", or "palm"
    """
    print(f"\n📷 Capturing {modality} image...")
    print("   Press SPACE to capture, ESC to cancel")
    
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        print("❌ Could not open camera")
        return False
    
    # Set camera resolution
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    captured = False
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Failed to capture frame")
            break
        
        # Show instructions on frame
        cv2.putText(frame, f"Capture {modality}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to capture", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Press ESC to cancel", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        cv2.imshow(f"Capture {modality}", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:  # SPACE key to capture
            save_path = f"input/{modality}.jpg"
            cv2.imwrite(save_path, frame)
            print(f"✅ {modality.capitalize()} captured and saved to: {save_path}")
            captured = True
            break
        elif key == 27:  # ESC key to cancel
            print(f"⚠️  {modality.capitalize()} capture cancelled")
            break
    
    cam.release()
    cv2.destroyAllWindows()
    
    # Small delay to ensure window closes
    cv2.waitKey(1)
    
    return captured

def upload_from_file(modality):
    """
    Upload image from file for a specific modality
    
    Args:
        modality: "face", "finger", or "palm"
    """
    print(f"\n📁 Uploading {modality} image...")
    
    # Create Tkinter root window and hide it
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title=f"Select {modality} image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
    )
    
    if file_path:
        if os.path.exists(file_path):
            # Read and save the image
            try:
                img = cv2.imread(file_path)
                if img is None:
                    print(f"❌ Could not read image: {file_path}")
                    return False
                
                save_path = f"input/{modality}.jpg"
                cv2.imwrite(save_path, img)
                print(f"✅ {modality.capitalize()} uploaded: {file_path}")
                print(f"   Saved to: {save_path}")
                return True
            except Exception as e:
                print(f"❌ Error saving image: {e}")
                return False
        else:
            print(f"❌ File does not exist: {file_path}")
            return False
    else:
        print(f"⚠️  No file selected for {modality}")
        return False

def capture_all_modalities():
    """Capture all three modalities"""
    create_input_folder()
    
    modalities = ["face", "finger", "palm"]
    
    print("\n" + "="*60)
    print("        CAPTURE ALL BIOMETRIC MODALITIES")
    print("="*60)
    
    for modality in modalities:
        while True:
            print(f"\n📸 {modality.upper()} CAPTURE")
            print("1. Use Camera")
            print("2. Upload from File")
            print("3. Skip this modality")
            
            choice = input("\nChoose option (1-3): ").strip()
            
            if choice == "1":
                if capture_from_camera(modality):
                    break
                else:
                    print("⚠️  Capture failed. Try again or choose another option.")
            elif choice == "2":
                if upload_from_file(modality):
                    break
                else:
                    print("⚠️  Upload failed. Try again or choose another option.")
            elif choice == "3":
                print(f"⚠️  Skipping {modality} capture")
                # Remove existing file if any
                if os.path.exists(f"input/{modality}.jpg"):
                    os.remove(f"input/{modality}.jpg")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    # Check which modalities were captured
    captured_modalities = []
    for modality in modalities:
        if os.path.exists(f"input/{modality}.jpg"):
            captured_modalities.append(modality)
    
    if captured_modalities:
        print(f"\n✅ Captured modalities: {', '.join(captured_modalities)}")
        return True
    else:
        print("\n❌ No modalities captured")
        return False

def verify_captured_images():
    """Verify the captured/uploaded images"""
    print("\n" + "="*60)
    print("       VERIFICATION IN PROGRESS")
    print("="*60)
    
    # Check which images exist
    available_images = []
    for modality in ["face", "finger", "palm"]:
        if os.path.exists(f"input/{modality}.jpg"):
            available_images.append(modality)
    
    if not available_images:
        print("❌ No images to verify")
        return
    
    print(f"📸 Available images: {', '.join(available_images)}")
    
    try:
        results = {}
        
        # Verify each available modality
        if "face" in available_images:
            face_person, face_score = verify_face("input/face.jpg")
            results["face"] = (face_person, face_score)
        
        if "finger" in available_images:
            finger_person, finger_score = verify_finger("input/finger.jpg")
            results["finger"] = (finger_person, finger_score)
        
        if "palm" in available_images:
            palm_person, palm_score = verify_palm("input/palm.jpg")
            results["palm"] = (palm_person, palm_score)
        
        # Display individual results
        print(f"\n📊 INDIVIDUAL MODALITY RESULTS:")
        for modality, (person, score) in results.items():
            print(f"   {modality.capitalize():<12} {person} ({score:.3f})")
        
        # Check if all agree
        if len(results) > 1:
            persons = [person for person, _ in results.values()]
            if all(p == persons[0] for p in persons):
                identified_person = persons[0]
                print(f"\n✅ All modalities agree: {identified_person}")
                
                # Get verification scores for this person
                verification_scores = {}
                if "face" in results:
                    verification_scores["face"] = verify_face("input/face.jpg", identified_person)
                if "finger" in results:
                    verification_scores["finger"] = verify_finger("input/finger.jpg", identified_person)
                if "palm" in results:
                    verification_scores["palm"] = verify_palm("input/palm.jpg", identified_person)
                
                # Fusion (only if we have at least 2 modalities)
                if len(verification_scores) >= 2:
                    scores_list = list(verification_scores.values())
                    
                    # Apply different fusion based on number of modalities
                    if len(scores_list) == 3:
                        # All three: face, finger, palm
                        final_score = fuse(scores_list[0], scores_list[1], scores_list[2])
                        weights = "(0.4, 0.3, 0.3)"
                    elif len(scores_list) == 2:
                        # Two modalities
                        if "face" in verification_scores and "finger" in verification_scores:
                            final_score = 0.6 * verification_scores["face"] + 0.4 * verification_scores["finger"]
                            weights = "(0.6, 0.4)"
                        elif "face" in verification_scores and "palm" in verification_scores:
                            final_score = 0.6 * verification_scores["face"] + 0.4 * verification_scores["palm"]
                            weights = "(0.6, 0.4)"
                        else:  # finger + palm
                            final_score = 0.5 * verification_scores["finger"] + 0.5 * verification_scores["palm"]
                            weights = "(0.5, 0.5)"
                    
                    print(f"\n🔐 VERIFICATION SCORES for {identified_person}:")
                    for modality, score in verification_scores.items():
                        print(f"   {modality.capitalize():<12} {score:.3f}")
                    
                    print(f"\n⚖️  FUSION (weights {weights}): {final_score:.3f}")
                    
                    # Decision
                    THRESHOLD = 0.85
                    print("\n" + "-"*60)
                    if final_score >= THRESHOLD:
                        print(f"✅ ACCESS GRANTED for {identified_person}")
                        print(f"   Confidence: {final_score:.1%}")
                        security_level = "HIGH" if final_score > 0.9 else "MEDIUM" if final_score > 0.8 else "LOW"
                        print(f"   Security Level: {security_level}")
                    else:
                        print(f"❌ ACCESS DENIED")
                        print(f"   Score {final_score:.3f} < Threshold {THRESHOLD}")
                
                else:
                    print("\n⚠️  Only one modality available - cannot perform fusion")
                    print("   Need at least 2 modalities for reliable verification")
            
            else:
                print("\n❌ MODALITIES DISAGREE - ACCESS DENIED")
                print("   Different persons detected:")
                for modality, (person, score) in results.items():
                    print(f"   {modality.capitalize()}: {person} ({score:.3f})")
                print("\n⚠️  Possible security threat or poor quality images")
        
        else:
            print("\n⚠️  Only one modality captured")
            modality, (person, score) = list(results.items())[0]
            print(f"   {modality.capitalize()} identifies: {person} ({score:.3f})")
            print("   Need at least 2 modalities for reliable verification")
    
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        print("Possible issues:")
        print("1. Corrupted image files")
        print("2. Images don't match expected format/size")
        print("3. Model loading error")
    
    print("="*60)

def run_evaluation():
    """Run the complete evaluation"""
    print("\n📊 Running system evaluation...")
    print("This will test all modalities and calculate accuracy metrics.")
    
    # Check if evaluation script exists
    if not os.path.exists("evaluation/evaluate_complete.py"):
        print("❌ Evaluation script not found")
        print("Please ensure 'evaluation/evaluate_complete.py' exists")
        return
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "evaluation/evaluate_complete.py"], 
                              capture_output=True, text=True)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(result.stdout)
        if result.stderr:
            print("ERRORS:")
            print(result.stderr)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

def clear_input_folder():
    """Clear the input folder"""
    if os.path.exists("input"):
        for file in os.listdir("input"):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                os.remove(f"input/{file}")
        print("✅ Input folder cleared")

def main():
    """Main application menu"""
    
    # Check models first
    if not check_models_exist():
        sys.exit(1)
    
    create_input_folder()
    
    while True:
        print("\n" + "="*60)
        print("           HYBRID MULTI-MODAL BIOMETRIC SYSTEM")
        print("="*60)
        print("1. 📸 Capture & Verify (Camera/Upload)")
        print("2. 🔄 Verify Existing Images (from input/ folder)")
        print("3. 📊 Run System Evaluation")
        print("4. 🧹 Clear Input Folder")
        print("5. 🚪 Exit")
        
        try:
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                # Capture new images and verify
                if capture_all_modalities():
                    verify_captured_images()
            
            elif choice == "2":
                # Verify existing images
                verify_captured_images()
            
            elif choice == "3":
                # Run evaluation
                run_evaluation()
                input("\nPress Enter to continue...")
            
            elif choice == "4":
                # Clear input folder
                clear_input_folder()
            
            elif choice == "5":
                print("\n👋 Goodbye! Thank you for using the system.")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-5.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Welcome message
    print("\n" + "="*60)
    print("        WELCOME TO HYBRID BIOMETRIC SYSTEM")
    print("="*60)
    print("System Features:")
    print("  • Face Recognition (MobileNetV2)")
    print("  • Fingerprint Recognition (Custom CNN)")
    print("  • Palm Recognition (MobileNetV2)")
    print("  • Weighted Score Fusion")
    print("  • Camera Capture & File Upload")
    print("="*60)
    
    main()

'''import os
import sys
from inference.verify_face import verify as verify_face
from inference.verify_finger import verify as verify_finger
from inference.verify_palm import verify as verify_palm
from fusion.weighted_fusion import weighted_fusion as fuse

def check_input_files():
    """Check if input images exist"""
    required = ["input/face.jpg", "input/finger.jpg", "input/palm.jpg"]
    missing = [f for f in required if not os.path.exists(f)]
    
    if missing:
        print("❌ Missing input images:")
        for f in missing:
            print(f"   - {f}")
        print("\nPlease:")
        print("1. Create 'input/' folder")
        print("2. Place face.jpg, finger.jpg, palm.jpg in it")
        print("3. Or use a UI tool to capture images")
        return False
    return True

def simple_verification():
    """Simple verification without UI"""
    print("="*60)
    print("       HYBRID MULTI-MODAL BIOMETRIC SYSTEM")
    print("="*60)
    
    if not check_input_files():
        return
    
    print("\n🔍 Identifying person...")
    
    try:
        # Step 1: Identify from each modality
        face_person, face_score = verify_face("input/face.jpg")
        finger_person, finger_score = verify_finger("input/finger.jpg")
        palm_person, palm_score = verify_palm("input/palm.jpg")
        
        print(f"\n📊 Individual Modality Results:")
        print(f"   Face:        {face_person} ({face_score:.3f})")
        print(f"   Fingerprint: {finger_person} ({finger_score:.3f})")
        print(f"   Palm:        {palm_person} ({palm_score:.3f})")
        
        # Step 2: Check if all agree
        if face_person == finger_person == palm_person:
            identified_person = face_person
            print(f"\n✅ All modalities agree: {identified_person}")
            
            # Step 3: Verify this specific person
            print(f"\n🔐 Verifying {identified_person}...")
            face_verify = verify_face("input/face.jpg", identified_person)
            finger_verify = verify_finger("input/finger.jpg", identified_person)
            palm_verify = verify_palm("input/palm.jpg", identified_person)
            
            # Step 4: Fusion
            final_score = fuse(face_verify, finger_verify, palm_verify)
            
            print(f"\n📈 Verification Scores:")
            print(f"   Face:        {face_verify:.3f}")
            print(f"   Fingerprint: {finger_verify:.3f}")
            print(f"   Palm:        {palm_verify:.3f}")
            print(f"   Fusion:      {final_score:.3f}")
            
            # Step 5: Decision
            THRESHOLD = 0.85
            print("\n" + "-"*60)
            if final_score >= THRESHOLD:
                print(f"✅ ACCESS GRANTED for {identified_person}")
                print(f"   Confidence: {final_score:.1%}")
                print(f"   Security Level: {'HIGH' if final_score > 0.9 else 'MEDIUM'}")
            else:
                print(f"❌ ACCESS DENIED")
                print(f"   Score {final_score:.3f} < Threshold {THRESHOLD}")
            
        else:
            print("\n❌ MODALITIES DISAGREE - ACCESS DENIED")
            print(f"   Face says: {face_person}")
            print(f"   Finger says: {finger_person}")
            print(f"   Palm says: {palm_person}")
            print("\n⚠️  Possible security threat or poor quality images")
            
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        print("Please ensure:")
        print("1. Models are trained (run training scripts first)")
        print("2. Images are valid JPG/PNG files")
        print("3. Input images are clear and properly captured")
    
    print("="*60)

def quick_evaluation():
    """Run quick evaluation"""
    print("\n📊 Running evaluation...")
    os.system("cd evaluation && python evaluate.py")
    input("\nPress Enter to continue...")

def main():
    """Main menu"""
    while True:
        print("\n" + "="*60)
        print("            HYBRID BIOMETRIC SYSTEM")
        print("="*60)
        print("1. Verify Person (using input/ folder images)")
        print("2. Run System Evaluation")
        print("3. Exit")
        
        try:
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == "1":
                simple_verification()
            elif choice == "2":
                quick_evaluation()
            elif choice == "3":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Check if trained models exist
    required_models = [
        "models/face_model.h5",
        "models/finger_model.h5", 
        "models/palm_model.h5",
        "models/face_classes.json"
    ]
    
    missing_models = [m for m in required_models if not os.path.exists(m)]
    
    if missing_models:
        print("❌ Missing trained models:")
        for m in missing_models:
            print(f"   - {m}")
        print("\nPlease train models first:")
        print("cd training && python train_face.py")
        print("cd training && python train_finger.py")
        print("cd training && python train_palm.py")
        sys.exit(1)
    
    main()'''