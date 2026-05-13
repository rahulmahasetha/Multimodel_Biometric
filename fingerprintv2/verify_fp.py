import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def verify_fp(query_emb, db, threshold=0.55):
    best_score = -1
    best_id = None

    for pid, ref_emb in db.items():
        score = cosine_similarity(query_emb, ref_emb)
        if score > best_score:
            best_score = score
            best_id = pid

    return best_score >= threshold, best_id, best_score
