import cv2
import numpy as np
import pickle
from pathlib import Path
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

database = {}
root = Path('/IntrudeDetector/allowed_faces')

for person_dir in root.iterdir():
    if not person_dir.is_dir():
        continue

    person = person_dir.name

    embeddings = []

    for image_path in person_dir.glob('*'):
        img = cv2.imread(str(image_path))

        if img is None:
            continue

        faces = app.get(img)

        if len(faces) != 1:
            print(
                f"{image_path}: "
                f"{len(faces)} visage(s), ignoré"
            )
            continue

        embedding = faces[0].embedding
        embedding /= np.linalg.norm(embedding) # normalization
        embeddings.append(embedding)

    if embeddings:
        mean = np.mean(embeddings, axis=0)
        mean /= np.linalg.norm(mean)
        database[person] = mean
        print(
            f"{person}: "
            f"{len(embeddings)} images")

with open('faces.pkl', 'wb') as f:
    pickle.dump(database, f)

print('Database created !')