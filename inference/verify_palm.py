import tensorflow as tf
import numpy as np
import json
from PIL import Image

class PalmVerifier:
    def __init__(self):
        """Load palm model and class mapping"""
        try:
            self.model = tf.keras.models.load_model("../models/palm_model.h5")
            with open("../models/palm_classes.json", "r") as f:
                self.classes = json.load(f)
            self.rev = {v: k for k, v in self.classes.items()}
            self.img_size = 224
            print("✅ Palm verifier loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load palm verifier: {e}")
            raise
    
    def verify(self, image_path, person_id=None):
        """Verify palm image"""
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
            print(f"❌ Palm verification error: {e}")
            if person_id:
                return 0.0
            else:
                return "unknown", 0.0

# Create global instance
palm_verifier = PalmVerifier()
verify = palm_verifier.verify