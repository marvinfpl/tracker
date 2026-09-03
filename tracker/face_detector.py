from insightface.app import FaceAnalysis
import cv2
import numpy as np
import pickle

app = FaceAnalysis(
    name='buffalo_l',
    providers=['CPUExecutionProvider'],
)

app.prepare(ctx_id=-1, det_size=(320, 320))

print('InsightFace Ready')

with open('faces.pkl', 'rb') as f:
    database = pickle.load(f)

print('database loaded')

THRESHOLD = 0.65

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize(embedding):
    best_name = 'Unknown'
    best_score = -1

    for name, known_embedding in database.items():
        score = cosine_similarity(embedding, known_embedding)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score < THRESHOLD:
        return 'Unknown', best_score

    return best_name, best_score


def recognize_face(frame):
    faces = app.get(frame)

    if len(faces) == 0:
        return frame, None

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        det_score = face.det_score
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if face.embedding is not None:
            name, r_score = recognize(face.embedding)

            cv2.putText(frame, f'Face {r_score:.2f} Name: {name}', (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
            print("Bounding box: ", face.bbox, "Score: ", face.det_score, "Embedding: ", face.embedding.shape)

    return frame

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_skip = 15
counter = 0
last_results = []

while True:
    counter += 1
    ok, frame = cap.read()
    if not ok:
        break

    if counter % frame_skip == 0:
        faces = app.get(frame)
        results = []

        for face in faces:
            if face.embedding is None:
                continue
            x1, y1, x2, y2 = map(int, face.bbox)
            name, recognition_score = recognize(face.embedding)
            results.append((x1, y1, x2, y2, name, recognition_score))
            last_results = results

    for (x1, y1, x2, y2, name, score) in last_results:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{name} {score:.2f}%', (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Test Recognize', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

    