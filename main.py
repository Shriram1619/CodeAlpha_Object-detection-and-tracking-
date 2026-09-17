!pip install ultralytics opencv-python

import cv2
from google.colab.output import eval_js
from PIL import Image
import io
import base64
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Start webcam UI
print("Click the 'Capture & Detect Frame' button below your camera feed to detect objects:")
data = eval_js('''
    async function runDetection() {
        const div = document.createElement('div');
        div.style.marginBottom = '10px';

        const video = document.createElement('video');
        video.style.display = 'block';
        video.style.width = '400px';
        video.style.borderRadius = '8px';

        const btn = document.createElement('button');
        btn.textContent = 'Capture & Detect Frame';
        btn.style.padding = '8px 16px';
        btn.style.marginTop = '10px';
        btn.style.cursor = 'pointer';

        const stream = await navigator.mediaDevices.getUserMedia({video: true});
        document.body.appendChild(div);
        div.appendChild(video);
        div.appendChild(btn);
        video.srcObject = stream;
        await video.play();

        return new Promise((resolve) => {
            btn.onclick = () => {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);
                // Stop the stream and remove the elements when frame is captured
                stream.getTracks().forEach(track => track.stop());
                div.remove();
                resolve(canvas.toDataURL('image/jpeg', 0.8));
            };
        });
    }
    runDetection(); // Call the function without 'return' keyword here
''')

# Convert captured image to OpenCV format
binary = base64.b64decode(data.split(',')[1])
image = Image.open(io.BytesIO(binary))
frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Run YOLO Object Detection & Tracking
results = model.track(frame_bgr, persist=True)

# Save output image
annotated_frame = results[0].plot()
output_path = "detected_frame.jpg"
cv2.imwrite(output_path, annotated_frame)

print("\n--- Detection Results ---")
print(f"Annotated frame saved to '{output_path}'! Check your files tab to open or download it.")
