from insightface.app import FaceAnalysis
import cv2
import numpy as np

app = FaceAnalysis(
    name='buffalo_l',
    providers=['CPUExecutionProvider'],
)

app.prepare(ctx_id=-1, det_size=(640, 640))

print('InsightFace Ready')

def detect_face(frame):
    faces = app.get(frame)

    if len(faces) == 0:
        return frame, None

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        score = face.det_score
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Face {score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
        print("Bounding box: ", face.bbox, "Score: ", face.det_score, "Embedding: ", face.embedding.shape)

    return frame, face.embedding