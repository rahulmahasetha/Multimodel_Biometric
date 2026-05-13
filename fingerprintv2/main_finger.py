import os
import cv2
import torch
import numpy as np
import time
from model_fp import FingerprintNet
from verify_fp import verify_fp

# ================= CONFIG =================
IMAGE_SIZE = 128
THRESHOLD = 0.55
MODEL_PATH = "fp_model.pth"

NUM_FRAMES = 20        # how many frames to capture
CAPTURE_TIME = 4      # seconds limit

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= LOAD MODEL =================
model = FingerprintNet().to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("✅ Loaded trained fingerprint model")
else:
    print("❌ fp_model.pth not found. Train model first.")
    exit()

model.eval()

# ================= LOAD TEMPLATES =================
db = np.load("fp_db.npy", allow_pickle=True).item()

# ================= PREPROCESS =================
def preprocess_fp(img):
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    return img

# ===================================================
# IMAGE PATH VERIFICATION
# ===================================================
def verify_image():
    while True:
        img_path = input("\n🖐️ Enter fingerprint image path (or 'back'): ").strip()

        if img_path.lower() == "back":
            return

        if not os.path.exists(img_path):
            print("❌ Image path does not exist")
            continue

        img = cv2.imread(img_path, 0)
        if img is None:
            print("❌ Failed to read image")
            continue

        img_t = preprocess_fp(img)

        with torch.no_grad():
            emb = model(img_t).cpu().numpy()[0]

        emb = emb / np.linalg.norm(emb)

        ok, pid, score = verify_fp(emb, db, threshold=THRESHOLD)

        print("\n================ RESULT ================")
        if ok:
            print("✅ ACCEPTED")
            print(f"🆔 Identity : {pid}")
            print(f"🔢 Score    : {score:.3f}")
        else:
            print("❌ REJECTED")
            print(f"🔢 Score    : {score:.3f}")
        print("========================================")

# ===================================================
# CAMERA VERIFICATION (MULTI-FRAME)
# ===================================================
def verify_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not available")
        return

    print("\n📷 Fingerprint Camera Mode")
    print("👉 Press 'c' ONCE to capture")
    print("👉 Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Fingerprint Camera", gray)

        key = cv2.waitKey(1) & 0xFF

        # ---------- MULTI-FRAME CAPTURE ----------
        if key == ord("c"):
            print("\n⏳ Capturing multiple frames...")
            embeddings = []
            start = time.time()

            while len(embeddings) < NUM_FRAMES and (time.time() - start) < CAPTURE_TIME:
                ret, frame = cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img_t = preprocess_fp(gray)

                with torch.no_grad():
                    emb = model(img_t).cpu().numpy()[0]
                    emb = emb / np.linalg.norm(emb)
                    embeddings.append(emb)

                cv2.imshow("Capturing...", gray)
                cv2.waitKey(1)

            if len(embeddings) < 5:
                print("⚠️ Not enough good frames. Try again.\n")
                continue

            # ---------- AGGREGATE ----------
            final_emb = np.mean(embeddings, axis=0)
            final_emb = final_emb / np.linalg.norm(final_emb)

            ok, pid, score = verify_fp(final_emb, db, threshold=THRESHOLD)

            print("\n================ RESULT (CAMERA) ================")
            print("⚠️ Camera fingerprint is LOW CONFIDENCE")
            if ok:
                print("✅ ACCEPTED (Demo)")
                print(f"🆔 Identity : {pid}")
            else:
                print("❌ REJECTED")
            print(f"🔢 Score    : {score:.3f}")
            print("=================================================\n")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ===================================================
# MAIN MENU
# ===================================================
def main():
    print("\n🔐 FINGERPRINT VERIFICATION SYSTEM")

    while True:
        print("\nChoose an option:")
        print("1️⃣ Verify using Image (Recommended)")
        print("2️⃣ Verify using Camera (Multi-frame demo)")
        print("0️⃣ Exit")

        choice = input("👉 Enter choice: ").strip()

        if choice == "1":
            verify_image()
        elif choice == "2":
            verify_camera()
        elif choice == "0":
            print("👋 Exiting system")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
