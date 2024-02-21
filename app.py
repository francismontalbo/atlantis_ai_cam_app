from flask import Flask, render_template, Response
import cv2
import torchvision.transforms as transforms
import torchvision
import torch

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

# Load pre-trained Faster R-CNN model
weights = "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1"
box_score_thresh = 0.9
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, 
                                                                           weights=weights, 
                                                                           box_score_thresh=box_score_thresh)
model.eval()

# Define the classes for COCO dataset
CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Function to perform object detection on frames
def detect_objects(frame):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(frame).unsqueeze(0)
    with torch.no_grad():
        prediction = model(image_tensor)
    for idx in range(len(prediction[0]['boxes'])):
        box = prediction[0]['boxes'][idx].cpu().numpy()
        label = CLASSES[prediction[0]['labels'][idx].item()]
        score = prediction[0]['scores'][idx].item()
        if score > 0.5:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {score}', (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return frame

# Function to generate video feed
def video_feed(camera):
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = detect_objects(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plant_cam_feed')
def plant_cam_feed():
    return Response(video_feed(plant_cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fish_cam_feed')
def fish_cam_feed():
    return Response(video_feed(fish_cam), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
