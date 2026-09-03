import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])

app.prepare(ctx_id=-1,det_size=(640, 640))

with open("faces.pkl", "rb") as f:
    database = pickle.load(f)

THRESHOLD = 0.65

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize(embedding):
    best_name = 'unknown'
    best_score = -1

    for name, known_embedding in database.items():
        score = cosine_similarity(embedding, known_embedding)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score < THRESHOLD:
        return 'Unknown', best_score

    return best_name, best_score