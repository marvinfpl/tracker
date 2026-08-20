from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import cv2
from tracker.yolo import detect_faces

app = FastAPI()
templates = Jinja2Templates(directory='interface/templates')

def generate_frame():
        cap = cv2.VideoCapture(0)

        while True:
            ok, frame = cap.read()

            if not ok:
                return

            frame = detect_faces(frame)

            ok, buffer = cv2.imencode('.jpg', frame)

            if not ok:
                continue

            frame_bytes = buffer.tobytes()

            yield (
               b"--frame\r\n"b"Content-Type: image/jpeg\r\n\r\n"
               + frame_bytes
               + b"\r\n"
            )


@app.get('/camera')
def show_camera():
    return StreamingResponse(
        generate_frame(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get('/')
def home(request: Request):
     return templates.TemplateResponse(
          'index.html',
          {'request': request}
     )