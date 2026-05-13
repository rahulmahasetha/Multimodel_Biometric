import os
import cv2
import time
import torch
import numpy as np
from model_p import PalmprintNet
from verify_p import verify_palm

# ================= CONFIG =================
IMAGE_SIZE = 128
NUM_FRAMES = 15          # auto-captured frames
CAPTURE_TIME = 4        # seconds
THRESHOLD = 0.6

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= LOAD MODEL =================
model = PalmprintNet().to(device)
model.load_state_dict(torch.load("palm_model.pth", map_location=device))
model.eval()

# ================= LOAD TEMPLATES =================
db = np.load("palm_db.npy", allow_pickle=True).item()

# ===================================================
# IMAGE VERIFICATION
# ===================================================
def verify_image():
    while True:
        path = input("\n🖐️ Palm image path (or 'back'): ").strip()

        if path.lower() == "back":
            return

        if not os.path.exists(path):
            print("❌ Image path does not exist")
            continue

        img = cv2.imread(path, 0)
        if img is None:
            print("❌ Failed to read image")
            continue

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            emb = model(img).cpu().numpy()[0]

        emb /= np.linalg.norm(emb)

        ok, pid, score = verify_palm(emb, db, threshold=THRESHOLD)

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
# CAMERA VERIFICATION (AUTO MULTI-FRAME)
# ===================================================
def verify_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not available")
        return

    print("\n📸 Palm camera started")
    print("👉 Place palm clearly in front of camera")
    print("👉 Press 'c' ONCE to verify")
    print("👉 Press 'q' to quit camera\n")

    display_text = ""
    display_until = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Show last result for 3 seconds
        if display_text and time.time() < display_until:
            cv2.putText(
                frame,
                display_text,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if "ACCEPTED" in display_text else (0, 0, 255),
                2,
            )

        cv2.imshow("Palmprint Verification", frame)
        key = cv2.waitKey(1) & 0xFF

        # ---------- START VERIFICATION ----------
        if key == ord("c"):
            print("\n⏳ Capturing palm frames...")
            embeddings = []
            start = time.time()

            while len(embeddings) < NUM_FRAMES and (time.time() - start) < CAPTURE_TIME:
                ret, frame = cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi = gray  # full frame (no detector for palm)

                try:
                    roi = cv2.resize(roi, (IMAGE_SIZE, IMAGE_SIZE))
                except:
                    continue

                roi = roi / 255.0
                roi = torch.tensor(roi).unsqueeze(0).unsqueeze(0).float().to(device)

                with torch.no_grad():
                    emb = model(roi).cpu().numpy()[0]

                emb /= np.linalg.norm(emb)
                embeddings.append(emb)

                cv2.putText(frame, "Capturing...", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow("Palmprint Verification", frame)
                cv2.waitKey(1)

            if len(embeddings) < 5:
                print("⚠️ Not enough good frames. Try again.")
                continue

            # ---------- AGGREGATE ----------
            final_emb = np.mean(embeddings, axis=0)
            final_emb /= np.linalg.norm(final_emb)

            ok, pid, score = verify_palm(final_emb, db, threshold=THRESHOLD)

            if ok:
                display_text = f"ACCEPTED | {pid} | {score:.2f}"
            else:
                display_text = f"REJECTED | {score:.2f}"

            display_until = time.time() + 3

            print("\n================ RESULT ================")
            if ok:
                print("✅ ACCEPTED")
                print(f"🆔 Identity : {pid}")
                print(f"🔢 Score    : {score:.3f}")
            else:
                print("❌ REJECTED")
                print(f"🔢 Score    : {score:.3f}")
            print("========================================\n")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ===================================================
# MAIN MENU
# ===================================================
def main():
    print("\n🔐 PALMPRINT VERIFICATION SYSTEM")

    while True:
        print("\nChoose an option:")
        print("1️⃣ Verify using Image")
        print("2️⃣ Verify using Camera")
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
