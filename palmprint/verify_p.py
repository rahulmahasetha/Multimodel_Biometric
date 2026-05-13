import numpy as np

def verify_palm(emb, db, threshold=0.6):
    best_score, best_id = -1, None
    for pid, temp in db.items():
        s = np.dot(emb, temp)
        if s > best_score:
            best_score, best_id = s, pid
    return best_score >= threshold, best_id, best_score
