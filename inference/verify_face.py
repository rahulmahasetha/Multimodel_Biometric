import tensorflow as tf
import numpy as np
import json
from PIL import Image

class FaceVerifier:
    def __init__(self):
        """Load face model and class mapping"""
        try:
            self.model = tf.keras.models.load_model("../models/face_model.h5")
            with open("../models/face_classes.json", "r") as f:
                self.classes = json.load(f)
            self.rev = {v: k for k, v in self.classes.items()}
            self.img_size = 224
            print("✅ Face verifier loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load face verifier: {e}")
            raise
    
    def verify(self, image_path, person_id=None):
        """
        Verify face image
        
        Args:
            image_path: Path to face image
            person_id: If provided, returns probability for this person
                     If None, returns (person_name, probability)
        
        Returns:
            If person_id given: probability (0-1)
            If person_id None: (person_name, probability)
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.img_size, self.img_size))
            arr = np.array(img) / 255.0
            arr = np.expand_dims(arr, 0)
            
            # Predict
            pred = self.model.predict(arr, verbose=0)[0]
            
            if person_id is not None:
                # Check specific person
                if person_id in self.classes:
                    return float(pred[self.classes[person_id]])
                else:
                    return 0.0  # Person not in training
            else:
                # Find best match
                i = np.argmax(pred)
                return self.rev[i], float(pred[i])
                
        except Exception as e:
            print(f"❌ Face verification error: {e}")
            if person_id:
                return 0.0
            else:
                return "unknown", 0.0

# Create global instance for easy import
face_verifier = FaceVerifier()
verify = face_verifier.verify