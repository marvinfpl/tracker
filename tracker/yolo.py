from ultralytics import YOLO
import cv2

model = YOLO('yolo11n.pt')


def detect_faces(frame):
        results = model(frame, verbose=False)

        for result in results:
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                #H, W = frame.shape[:2]
                #image_cx = W / 2
                #image_cy = H / 2

                #cx = (x1 + x2) / 2
                #cy = (y1 + y2) / 2
        
                #error_x = cx - image_cx
                #error_y = cy - image_cy

        return frame