import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def evaluate_scores(scores, labels, threshold):
    preds = [1 if s >= threshold else 0 for s in scores]

    correct = sum(p == y for p, y in zip(preds, labels))
    acc = correct / len(labels)

    FAR = sum(p == 1 and y == 0 for p, y in zip(preds, labels)) / max(1, sum(l == 0 for l in labels))
    FRR = sum(p == 0 and y == 1 for p, y in zip(preds, labels)) / max(1, sum(l == 1 for l in labels))

    return acc * 100, FAR * 100, FRR * 100
