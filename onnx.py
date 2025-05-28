import cv2
import numpy as np
import onnxruntime

# Initialize ONNX Runtime session
session = onnxruntime.InferenceSession(r'runs\detect\train\weights\best.onnx', providers=['CPUExecutionProvider'])

# Get model input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_name = session.get_outputs()[0].name

webcamera = cv2.VideoCapture(0)

while True:
    success, frame = webcamera.read()
    if not success:
        break

    # Preprocess image
    img = cv2.resize(frame, (640, 640))
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Run inference
    results = session.run([output_name], {input_name: img})[0]

    # Post-process results
    boxes = results[0]  # Assuming output format is similar to YOLO
    valid_detections = boxes[boxes[:, 4] > 0.8]  # Filter by confidence threshold

    # Draw boxes on frame
    for box in valid_detections:
        x1, y1, x2, y2 = map(int, box[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"Total: {len(valid_detections)}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Live Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

webcamera.release()
cv2.destroyAllWindows()