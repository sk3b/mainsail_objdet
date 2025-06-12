import cv2
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import numpy as np
import time
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    print("Shutting down and releasing webcam")
    if cap.isOpened():
        cap.release()

app = FastAPI(lifespan=lifespan)

# Initialize YOLO model
model = YOLO("best.mnn")
labels = model.names

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200

def generate_frames():
    try:
        while True:
            t_start = time.perf_counter()

            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Perform object detection
            results = model(frame, verbose=False)
            detections = results[0].boxes
            object_count = 0

            for i in range(len(detections)):
                # Get bounding box coordinates
                xyxy_tensor = detections[i].xyxy.cpu()
                xyxy = xyxy_tensor.numpy().squeeze()
                xmin, ymin, xmax, ymax = xyxy.astype(int)

                # Get bounding box class ID and name
                classidx = int(detections[i].cls.item())
                classname = labels[classidx]

                # Get bounding box confidence
                conf = detections[i].conf.item()

                # Draw box if confidence threshold is high enough
                if conf > 0.5:
                    color = bbox_colors[classidx % 10]
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                    label = f'{classname}: {int(conf*100)}%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, labelSize[1] + 10)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10),
                                 (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1)

                    object_count += 1

            # Calculate FPS
            t_stop = time.perf_counter()
            frame_rate_calc = float(1 / (t_stop - t_start))

            if len(frame_rate_buffer) >= fps_avg_len:
                frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)

            # Calculate average FPS
            avg_frame_rate = np.mean(frame_rate_buffer)

            # Draw FPS on frame
            cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 155), 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                continue
            frame_bytes = buffer.tobytes()

            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        print("Releasing webcam")
        cap.release()

@app.get("/")
async def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            img { 
                max-width: 100%; 
                height: auto;
            }
        </style>
    </head>
    <body>
        <div>
            <img src="/video_feed" alt="Video Stream">
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6969)