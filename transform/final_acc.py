import os
import sys
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from insightface.app import FaceAnalysis

# ================= PATH =================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from fingerprint.model_fp import FingerprintNet
from palmprint.model_p import PalmprintNet
from transform.transformer_feature_fusion import TransformerFeatureFusion

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= DATA PATH =================
DATA_ROOT = os.path.join(ROOT, "Data")
FACE_DIR = os.path.join(DATA_ROOT, "augmented_face")
FP_DIR   = os.path.join(DATA_ROOT, "fingerprint_augmented")
PALM_DIR = os.path.join(DATA_ROOT, "palmprint_augmented")

ALL_USERS = sorted(os.listdir(FACE_DIR))

# -------- Identity Split --------
TRAIN_USERS = ALL_USERS[:350]
TEST_USERS  = ALL_USERS[350:]

print("Train identities:", len(TRAIN_USERS))
print("Test identities :", len(TEST_USERS))

# ================= LOAD MODELS =================
print("\nLoading models...")

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640,640))

fp_model = FingerprintNet().to(device)
fp_model.load_state_dict(
    torch.load(os.path.join(ROOT,"fingerprint","fp_model.pth"),
               map_location=device)
)
fp_model.eval()

palm_model = PalmprintNet().to(device)
palm_model.load_state_dict(
    torch.load(os.path.join(ROOT,"palmprint","palm_model.pth"),
               map_location=device)
)
palm_model.eval()

fusion_model = TransformerFeatureFusion().to(device)
fusion_model.load_state_dict(
    torch.load(os.path.join(ROOT,"transform","fusion_transformer.pth"),
               map_location=device)
)
fusion_model.eval()

print("Models loaded successfully.\n")

# ================= HELPER FUNCTIONS =================
def get_face_emb(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_app.get(img)
    if not faces:
        return None
    return torch.tensor(faces[0].normed_embedding).float().to(device)

def get_fp_emb(path):
    img = cv2.imread(path, 0)
    if img is None:
        return None
    img = cv2.resize(img,(128,128))/255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        e = fp_model(t)[0]
    return e / torch.norm(e)

def get_palm_emb(path):
    img = cv2.imread(path, 0)
    if img is None:
        return None
    img = cv2.resize(img,(128,128))/255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        e = palm_model(t)[0]
    return e / torch.norm(e)

def fuse(f,fp,p):
    with torch.no_grad():
        return fusion_model(
            f.unsqueeze(0),
            fp.unsqueeze(0),
            p.unsqueeze(0)
        )[0]

def cosine(a,b):
    return torch.cosine_similarity(
        a.unsqueeze(0),b.unsqueeze(0)
    ).item()

# ================= BUILD ENROLLMENT (TEST USERS ONLY) =================
print("Building enrollment templates...")
enroll_db = {}

for uid in TEST_USERS:
    fdir = os.path.join(FACE_DIR,uid)
    fpdir = os.path.join(FP_DIR,uid)
    pdir = os.path.join(PALM_DIR,uid)

    if not os.path.isdir(fdir):
        continue

    files = sorted(os.listdir(fdir))
    if len(files)==0:
        continue

    f = get_face_emb(os.path.join(fdir,files[0]))
    fp = get_fp_emb(os.path.join(fpdir,files[0]))
    p = get_palm_emb(os.path.join(pdir,files[0]))

    if f is None or fp is None or p is None:
        continue

    enroll_db[uid] = fuse(f,fp,p)

print("Enrollment users:",len(enroll_db))

# ================= MULTI-PROBE EVALUATION =================
genuine_scores=[]
impostor_scores=[]
correct_id=0
total_id=0

print("\nRunning multi-probe evaluation...")

for uid in tqdm(enroll_db.keys()):

    fdir = os.path.join(FACE_DIR,uid)
    fpdir = os.path.join(FP_DIR,uid)
    pdir = os.path.join(PALM_DIR,uid)

    files = sorted(os.listdir(fdir))

    # use ALL remaining images as probes
    for i in range(1,len(files)):

        f = get_face_emb(os.path.join(fdir,files[i]))
        fp = get_fp_emb(os.path.join(fpdir,files[i]))
        p = get_palm_emb(os.path.join(pdir,files[i]))

        if f is None or fp is None or p is None:
            continue

        probe = fuse(f,fp,p)

        # verification scores
        for other_uid in enroll_db:
            score = cosine(probe,enroll_db[other_uid])

            if other_uid==uid:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

        # identification (Rank-1)
        scores={u:cosine(probe,enroll_db[u]) for u in enroll_db}
        pred=max(scores,key=scores.get)
        total_id+=1
        if pred==uid:
            correct_id+=1

# ================= METRICS =================
genuine_scores=np.array(genuine_scores)
impostor_scores=np.array(impostor_scores)

all_scores=np.concatenate([genuine_scores,impostor_scores])
thresholds=np.linspace(all_scores.min(),all_scores.max(),3000)

best_acc=0
best_th=0
eer=0
eer_th=0
min_diff=1e9

total_trials=len(genuine_scores)+len(impostor_scores)

for th in thresholds:

    far=np.sum(impostor_scores>=th)/len(impostor_scores)
    frr=np.sum(genuine_scores<th)/len(genuine_scores)

    acc=(
        np.sum(genuine_scores>=th)+
        np.sum(impostor_scores<th)
    )/total_trials

    if acc>best_acc:
        best_acc=acc
        best_th=th

    if abs(far-frr)<min_diff:
        min_diff=abs(far-frr)
        eer=(far+frr)/2
        eer_th=th

# AUC
labels=np.concatenate([
    np.ones(len(genuine_scores)),
    np.zeros(len(impostor_scores))
])
scores=np.concatenate([genuine_scores,impostor_scores])
auc=roc_auc_score(labels,scores)

# ================= REPORT =================
print("\n===== JOURNAL-GRADE REPORT =====")
print("Test Users        :",len(enroll_db))
print("Genuine Trials    :",len(genuine_scores))
print("Impostor Trials   :",len(impostor_scores))
print("--------------------------------")
print("Genuine mean      :",np.mean(genuine_scores))
print("Genuine std       :",np.std(genuine_scores))
print("Impostor mean     :",np.mean(impostor_scores))
print("Impostor std      :",np.std(impostor_scores))
print("--------------------------------")
print("Best Threshold    :",round(best_th,4))
print("Verification Acc  :",round(best_acc*100,3),"%")
print("EER               :",round(eer*100,4),"%")
print("EER Threshold     :",round(eer_th,4))
print("AUC               :",round(auc,6))
print("Rank-1 Accuracy   :",round((correct_id/total_id)*100,3),"%")
print("================================")



'''import os
import sys
import cv2
import numpy as np
import torch
from tqdm import tqdm
from insightface.app import FaceAnalysis

# ================= PATH SETUP =================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from fingerprint.model_fp import FingerprintNet
from palmprint.model_p import PalmprintNet
from transform.transformer_feature_fusion import TransformerFeatureFusion

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_ROOT = os.path.join(ROOT, "Data")
FACE_DIR = os.path.join(DATA_ROOT, "augmented_face")
FP_DIR   = os.path.join(DATA_ROOT, "fingerprint_augmented")
PALM_DIR = os.path.join(DATA_ROOT, "palmprint_augmented")

# ================= IDENTITY SPLIT =================
ALL_USERS = sorted([u for u in os.listdir(FACE_DIR) if u.isdigit()],
                   key=lambda x: int(x))

TRAIN_USERS = ALL_USERS[:350]
TEST_USERS  = ALL_USERS[350:]

print("Train identities:", len(TRAIN_USERS))
print("Test identities :", len(TEST_USERS))

# ================= LOAD MODELS =================
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

fp_model = FingerprintNet().to(device)
fp_model.load_state_dict(
    torch.load(os.path.join(ROOT,"fingerprint","fp_model.pth"),
               map_location=device))
fp_model.eval()

palm_model = PalmprintNet().to(device)
palm_model.load_state_dict(
    torch.load(os.path.join(ROOT,"palmprint","palm_model.pth"),
               map_location=device))
palm_model.eval()

fusion_model = TransformerFeatureFusion().to(device)
fusion_model.load_state_dict(
    torch.load(os.path.join(ROOT,"transform","fusion_transformer.pth"),
               map_location=device))
fusion_model.eval()

# ================= HELPERS =================
def cosine(a, b):
    return torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def get_face(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_app.get(img)
    if not faces:
        return None
    return torch.tensor(faces[0].normed_embedding).float().to(device)

def get_fp(path):
    img = cv2.imread(path, 0)
    if img is None:
        return None
    img = cv2.resize(img,(128,128))/255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        e = fp_model(t)[0]
    return e/torch.norm(e)

def get_palm(path):
    img = cv2.imread(path, 0)
    if img is None:
        return None
    img = cv2.resize(img,(128,128))/255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        e = palm_model(t)[0]
    return e/torch.norm(e)

def fuse(f,fp,p):
    with torch.no_grad():
        return fusion_model(f.unsqueeze(0),
                            fp.unsqueeze(0),
                            p.unsqueeze(0))[0]

# ================= BUILD ENROLLMENT (TEST ONLY) =================
print("\nBuilding enrollment (test identities only)...")

enroll_db = {}

for uid in TEST_USERS:

    fdir = os.path.join(FACE_DIR, uid)
    fpdir = os.path.join(FP_DIR, uid)
    pdir = os.path.join(PALM_DIR, uid)

    files_f = sorted(os.listdir(fdir))
    if len(files_f) < 2:
        continue

    f = get_face(os.path.join(fdir, files_f[0]))
    fp = get_fp(os.path.join(fpdir, sorted(os.listdir(fpdir))[0]))
    p = get_palm(os.path.join(pdir, sorted(os.listdir(pdir))[0]))

    if f is None:
        continue
    if f is None or fp is None or p is None:
        continue

    enroll_db[uid] = fuse(f,fp,p)

print("Enrollment users:", len(enroll_db))
print("Valid enrolled test users:", len(enroll_db))
print("Skipped users:", len(TEST_USERS) - len(enroll_db))

# ================= EVALUATION =================
genuine_scores = []
impostor_scores = []
correct_id = 0
total_id = 0

print("\nRunning research evaluation...")

for uid in tqdm(enroll_db.keys()):

    fdir = os.path.join(FACE_DIR, uid)
    fpdir = os.path.join(FP_DIR, uid)
    pdir = os.path.join(PALM_DIR, uid)

    files_f = sorted(os.listdir(fdir))

    f = get_face(os.path.join(fdir, files_f[1]))
    fp = get_fp(os.path.join(fpdir, sorted(os.listdir(fpdir))[1]))
    p = get_palm(os.path.join(pdir, sorted(os.listdir(pdir))[1]))

    if f is None:
        continue
    if f is None or fp is None or p is None:
        continue

    probe = fuse(f,fp,p)

    scores = {}

    for other in enroll_db:
        s = cosine(probe, enroll_db[other])
        scores[other] = s

        if other == uid:
            genuine_scores.append(s)
        else:
            impostor_scores.append(s)

    predicted = max(scores, key=scores.get)
    total_id += 1
    if predicted == uid:
        correct_id += 1

# ================= METRICS =================
# ================= METRICS =================
genuine_scores = np.array(genuine_scores)
impostor_scores = np.array(impostor_scores)

all_scores = np.concatenate([genuine_scores, impostor_scores])
thresholds = np.linspace(all_scores.min(), all_scores.max(), 2000)

best_acc = 0
best_th = 0
eer = 0
eer_th = 0
min_diff = float("inf")

total_trials = len(genuine_scores) + len(impostor_scores)

for th in thresholds:

    far = np.sum(impostor_scores >= th) / len(impostor_scores)
    frr = np.sum(genuine_scores < th) / len(genuine_scores)

    acc = (
        np.sum(genuine_scores >= th) +
        np.sum(impostor_scores < th)
    ) / total_trials

    # Best accuracy threshold
    if acc > best_acc:
        best_acc = acc
        best_th = th

    # EER computation
    if abs(far - frr) < min_diff:
        min_diff = abs(far - frr)
        eer = (far + frr) / 2
        eer_th = th

print("\n===== RESEARCH REPORT =====")
print("Test Users        :", len(enroll_db))
print("Genuine Trials    :", len(genuine_scores))
print("Impostor Trials   :", len(impostor_scores))
print("--------------------------------")
print("Genuine mean      :", np.mean(genuine_scores))
print("Impostor mean     :", np.mean(impostor_scores))
print("Max impostor      :", np.max(impostor_scores))
print("Min genuine       :", np.min(genuine_scores))
print("--------------------------------")
print("Best Threshold    :", round(best_th, 4))
print("Verification Acc  :", round(best_acc * 100, 3), "%")
print("EER               :", round(eer * 100, 3), "%")
print("EER Threshold     :", round(eer_th, 4))
print("Rank-1 Accuracy   :", round((correct_id/total_id) * 100, 3), "%")
print("================================")
'''