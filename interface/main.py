from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import cv2
from tracker.yolo import detect_faces

app = FastAPI()
templates = Jinja2Templates(directory='interface/templates')
app.mount(
    '/static',
    StaticFiles(directory='interface/templates'),
    name='static',
)

stream = False

def generate_frame():
        global stream
        cap = cv2.VideoCapture(0)
        try:
            while stream:
                ok, frame = cap.read()

                if not ok:
                   break

                frame = detect_faces(frame)
                ok, buffer = cv2.imencode('.jpg', frame)

                if not ok:
                    continue

                yield (
                b"--frame\r\n"b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
                )

        finally:
             cap.release()


@app.get('/start_camera')
def start_camera():
    global stream
    stream = True
    return StreamingResponse(
        generate_frame(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get('/stop_camera')
def stop_camera():
    global stream
    stream = False
    return {'status': 'stopped'}

@app.get('/')
def home(request: Request):
     return templates.TemplateResponse(
          'index.html',
          {'request': request}
     )