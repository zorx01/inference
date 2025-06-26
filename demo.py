from flask import Flask, Response, render_template_string
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # 0 for Pi camera or USB webcam

# HTML page
HTML = """
<!DOCTYPE html>
<html>
<head><title>Raspberry Pi Camera</title></head>
<body>
<h1>Live Video Stream</h1>
<img src="{{ url_for('video_feed') }}">
</body>
</html>
"""

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
