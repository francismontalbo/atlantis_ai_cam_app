from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Camera indexes
PLANT_CAM_INDEX = 0
FISH_CAM_INDEX = 1

# Video capture objects
plant_cam = cv2.VideoCapture(PLANT_CAM_INDEX)
fish_cam = cv2.VideoCapture(FISH_CAM_INDEX)

def gen_frames(camera):
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plant_cam_feed')
def plant_cam_feed():
    return Response(gen_frames(plant_cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fish_cam_feed')
def fish_cam_feed():
    return Response(gen_frames(fish_cam), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
