import tensorflow as tf
import numpy as np
import json
from PIL import Image

class FingerVerifier:
    def __init__(self):
        """Load fingerprint model and class mapping"""
        try:
            self.model = tf.keras.models.load_model("../models/finger_model.h5")
            with open("../models/finger_classes.json", "r") as f:
                self.classes = json.load(f)
            self.rev = {v: k for k, v in self.classes.items()}
            self.img_size = 128
            print("✅ Fingerprint verifier loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load fingerprint verifier: {e}")
            raise
    
    def verify(self, image_path, person_id=None):
        """Verify fingerprint image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.img_size, self.img_size))
            arr = np.array(img) / 255.0
            arr = np.expand_dims(arr, 0)
            
            pred = self.model.predict(arr, verbose=0)[0]
            
            if person_id is not None:
                if person_id in self.classes:
                    return float(pred[self.classes[person_id]])
                else:
                    return 0.0
            else:
                i = np.argmax(pred)
                return self.rev[i], float(pred[i])
                
        except Exception as e:
            print(f"❌ Fingerprint verification error: {e}")
            if person_id:
                return 0.0
            else:
                return "unknown", 0.0

# Create global instance
finger_verifier = FingerVerifier()
verify = finger_verifier.verify