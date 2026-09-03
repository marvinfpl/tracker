from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import cv2
from tracker.face_detector import detect_face
from tracker.recognition import recognize

app = FastAPI()
templates = Jinja2Templates(directory='interface/templates')
app.mount(
    '/static',
    StaticFiles(directory='interface/templates'),
    name='static',
)

stream = False
current_name = 'Unknown'
current_score = 0.0
alarm_active = False

def generate_frame():
        global stream, current_name, current_score, alarm_active

        cap = cv2.VideoCapture(0)
        try:
            while stream:
                ok, frame = cap.read()

                if not ok:
                   break

                frame, embedding = detect_face(frame)
                if embedding is not None:
                    current_name, current_score = recognize(embedding)
                    if current_name == 'Unknown':
                        alarm_active = True
                    else:
                        print_info(current_score, current_name)
                     
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

@app.get('/stop_alarm')
def stop_alarm():
    global alarm_active
    alarm_active = False

@app.get('/alarm_status')
def alarm_status():
    return {'active': alarm_active}

@app.get('/print_info') 
def print_info():
    return {
        'name': current_name,
        'score': current_score,
    }