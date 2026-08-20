from ultralytics import YOLO
import cv2

model = YOLO('yolo11n.pt')
#model.track() for tracking

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print('cant receive any frame... ')
        break

    H, W = frame.shape[:2]
    image_cx = W / 2
    image_cy = H / 2


    results = model(frame)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
        
            error_x = cx - image_cx
            error_y = cy - image_cy


    cv2.imshow('Camera Output: ', frame)

    if 0xFF & cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()