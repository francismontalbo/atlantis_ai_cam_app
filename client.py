import base64
import cv2
from atlantis_edge.actuator_automation.ai_command_module import Observable
import torchvision.transforms as transforms
import torchvision
import torchvision.models.detection as detection_models
import torch
import time
import pygame
import socketio
import os

# observables for fish size 
class AIFishSize(Observable):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(AIFishSize, cls).__new__(cls)
        return cls.instance
            
    def set_fish_size(self, fish_size: str):
        self.fish_size = fish_size
        self.notify_observers()

# observables for future use when the ai needs to send commands to turn on/off actuators
class AICommand(Observable):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(AICommand, cls).__new__(cls)
        return cls.instance
            
    def send_command(self, actuator_name: str, duration: float):
        self.actuator_name = actuator_name
        self.duration = duration
        self.notify_observers()

PLANT_CAM_INDEX = 0
FISH_CAM_INDEX = 1
FRAME_RATE = 60
FRAME_WIDTH = 320
FRAME_HEIGHT = 320
SHOW_FPS = True

pygame.mixer.init()
sick_plant_sound = pygame.mixer.Sound("static/fx/sick_plant.mp3")
sick_fish_sound = pygame.mixer.Sound("static/fx/sick_fish.mp3")

plant_sound_interval = 10
fish_sound_interval = 10

plant_cam_counter = 0
fish_cam_counter = 0

last_played_time_plant = time.time()
last_played_time_fish = time.time()
connectedClient = []

aiFishSize = AIFishSize()
aiCommand = AICommand()
_current_fish_size = ""

def initialize_camera(
    index, frame_rate=FRAME_RATE, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT
):
    camera = cv2.VideoCapture(index)
    camera.set(cv2.CAP_PROP_FPS, frame_rate)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    return camera


plant_cam = initialize_camera(PLANT_CAM_INDEX)
fish_cam = initialize_camera(FISH_CAM_INDEX)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# weights = "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1"
# weights = torch.load(
#     os.path.join(os.path.dirname(__file__), os.path.abspath("weights/fish_model.pth")),
#     map_location=torch.device("cpu"),
# )
# weights_plant_model = "weights/plant_model.pth"
# model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
#     pretrained=True, weights=weights, box_score_thresh=box_score_thresh
# )

box_score_thresh = 0.9

# FISH MODEL
fish_model = detection_models.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)

fish_num_classes = 2
fish_in_features = fish_model.roi_heads.box_predictor.cls_score.in_features
fish_model.roi_heads.box_predictor = detection_models.faster_rcnn.FastRCNNPredictor(fish_in_features, fish_num_classes)

fish_model.load_state_dict(torch.load('weights/fish_model.pth', map_location=device))
fish_model = fish_model.to(device)
fish_model.eval()

# PLANT MODEL
plant_model = detection_models.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)

plant_num_classes = 2
plant_in_features = plant_model.roi_heads.box_predictor.cls_score.in_features
plant_model.roi_heads.box_predictor = detection_models.faster_rcnn.FastRCNNPredictor(plant_in_features, plant_num_classes)

plant_model.load_state_dict(torch.load('weights/fish_model.pth', map_location=device))
plant_model = plant_model.to(device)
plant_model.eval()


def load_classes(file_path):
    with open(file_path, "r") as file:
        classes = [line.strip() for line in file.readlines() if line.strip()]
    return classes


fish_classes_file = "fish_classes.txt"
plant_classes_file = "plant_classes.txt"
FISH_CLASSES = load_classes(fish_classes_file)
PLANT_CLASSES = load_classes(plant_classes_file)


# Function to perform object detection on frames
def detect_objects(frame, camera_index: int, sio: socketio.SimpleClient):
    global plant_cam_counter, fish_cam_counter, last_played_time_plant, last_played_time_fish

    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        fish_prediction = fish_model(image_tensor)
        plant_prediction = plant_model(image_tensor)

    plants_counted = 0
    fish_counted = 0
    current_fish_size = None
    for idx in range(len(fish_prediction[0]["boxes"])):
        box = fish_prediction[0]["boxes"][idx].cpu().numpy()
        fish_label = FISH_CLASSES[fish_prediction[0]["labels"][idx].item()]
        plant_label = PLANT_CLASSES[plant_prediction[0]["labels"][idx].item()]
        score = fish_prediction[0]["scores"][idx].item()

        if score > 0.9:
            cv2.rectangle(
                frame,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 255, 0),
                2,
            )
            score_formatted = "{:.2f}".format(score * 100)
            cv2.putText(
                frame,
                f"{fish_label}: {score_formatted}%",
                (int(box[0]), int(box[1])),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

            # label == ["healthy", "unhealthy"]
            if plant_label == "unhealthy":  
                plants_counted += 1
            # label in ["small_fish", "medium_fish", "big_fish"]
            if fish_label == "mediumfish":  
                current_fish_size = "mediumfish"
            elif fish_label == "smallfish":
                current_fish_size = "smallfish"
            elif fish_label == "bigfish":
                current_fish_size = "bigfish"
                
            aiFishSize.set_fish_size(current_fish_size)
            _current_fish_size = current_fish_size

    if camera_index == PLANT_CAM_INDEX:
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (220, 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(
            frame,
            f"Sick plants detected: {plants_counted}",
            (10, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if camera_index == FISH_CAM_INDEX:
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (220, 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(
            frame,
            f"Average Size of Fish: {current_fish_size}",
            (10, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if plants_counted > plant_cam_counter:
        sio.emit("play_sound", {"sound_url": "static/fx/sick_plant.mp3"})

    if fish_counted > fish_cam_counter:
        sio.emit("play_sound", {"sound_url": "static/fx/sick_fish.mp3"})

    plant_cam_counter = plants_counted
    fish_cam_counter = fish_counted

    return frame


def video_feed(camera: cv2.VideoCapture, camera_index: int, sio: socketio.SimpleClient):
    start_time = time.time()
    frame_count = 0

    while True:
        success, frame = camera.read()

        if not success:
            break

        # Perform object detection
        frame = detect_objects(frame, camera_index, sio)
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        if SHOW_FPS:
            height, width, _ = frame.shape
            position = (width - 100, 30)
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                position,
                cv2.FONT_HERSHEY_DUPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        # Apply compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # adjust quality as needed
        _, compressed_frame = cv2.imencode(".jpg", frame, encode_param)

        # Convert to bytes and send to server
        frame_bytes = compressed_frame.tobytes()
        sio.emit(
            "video_feed",
            {
                "frame": base64.b64encode(frame_bytes).decode("utf-8"),
                "camera_index": camera_index,
            },
        )

def get_current_fish_size():
    return _current_fish_size

with socketio.SimpleClient() as sio:
    print("Connecting to server")
    sio.connect("http://122.53.28.51:8000")
    # sio.connect("http://127.0.0.1:8000")
    print("Connected, starting feed")
    video_feed(plant_cam, FISH_CAM_INDEX, sio)
    print("Feed stopped")

sio2 = sio.client
sio2.connect("http://122.53.28.51:8200")


@sio2.on("viewer_connected")
def handle_viewer_connect(json):
    print(f"New client: {json['viewer_uuid']}")
    connectedClient.append(json["viewer_uuid"])
    print(f"Active clients: {len(connectedClient)}")


@sio2.on("viewer_disconnecting")
def handle_viewer_disconnect(json):

    if json["viewer_uuid"] in connectedClient:
        connectedClient.remove(json["viewer_uuid"])
        print(f"Removed client: {json['viewer_uuid']}")

    if len(connectedClient) == 0:
        print("No active clients...")
