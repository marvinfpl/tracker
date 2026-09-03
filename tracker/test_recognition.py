import cv2
import numpy as np
import pickle
from face_detector import detect_face

with open('faces.pkl', 'rb') as f:
    database = pickle.load(f)

print(database)

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

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame, embedding = detect_face(frame)
    if embedding is not None:
        name, score = recognize(embedding)
        if name == 'Unknown':
            print('No one has been recognized!')
        print(f'Name: {name}, Score: {score} \n')

    cv2.putText(frame, f'Name: {name}', cv2.FONT_HERSHEY_COMPLEX, )
    cv2.imshow('Test Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()