from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

PLANT_CAM_INDEX = 0
FISH_CAM_INDEX = 1

frame_rate = 15
frame_width = 640
frame_height = 480

def initialize_camera(index, frame_rate=frame_rate, frame_width=frame_width, frame_height=480):
    camera = cv2.VideoCapture(index)
    camera.set(cv2.CAP_PROP_FPS, frame_rate)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    return camera


plant_cam = initialize_camera(PLANT_CAM_INDEX)
fish_cam = initialize_camera(FISH_CAM_INDEX)

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
    app.run(debug=False)