import base64
import cv2
import torchvision.transforms as transforms
import torchvision
import torch
import time
import pygame
import socketio


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

weights = "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1"
box_score_thresh = 0.9
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    pretrained=True, weights=weights, box_score_thresh=box_score_thresh
)
model = model.to(device)
model.eval()


def load_classes(file_path):
    with open(file_path, "r") as file:
        classes = [line.strip() for line in file.readlines() if line.strip()]
    return classes


classes_file = "classes.txt"
CLASSES = load_classes(classes_file)


# Function to perform object detection on frames
def detect_objects(frame, camera_index: int, sio: socketio.SimpleClient):
    global plant_cam_counter, fish_cam_counter, last_played_time_plant, last_played_time_fish

    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image_tensor)

    if camera_index == PLANT_CAM_INDEX:
        plant_cam_counter = 0
    elif camera_index == FISH_CAM_INDEX:
        fish_cam_counter = 0

    sound_playing = False
    for idx in range(len(prediction[0]["boxes"])):
        box = prediction[0]["boxes"][idx].cpu().numpy()
        label = CLASSES[prediction[0]["labels"][idx].item()]
        score = prediction[0]["scores"][idx].item()
        if score > 0.5:
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
                f"{label}: {score_formatted}%",
                (int(box[0]), int(box[1])),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
            if label == "person":
                if camera_index == PLANT_CAM_INDEX:
                    plant_cam_counter += 1
                # elif camera_index == FISH_CAM_INDEX:
                #     fish_cam_counter += 1
                if not sound_playing:
                    sound_playing = True
                    if camera_index == PLANT_CAM_INDEX:
                        sio.emit(
                            "play_sound", {"sound_url": "static/fx/sick_plant.mp3"}
                        )
                    # elif camera_index == FISH_CAM_INDEX:
                    #     sio.emit('play_sound', {'sound_url': 'static/fx/sick_fish.mp3'})
            if label == "chair":
                if camera_index == FISH_CAM_INDEX:
                    fish_cam_counter += 1
                if not sound_playing:
                    sound_playing = True
                    sio.emit(
                        "play_sound", {"sound_url": "static/fx/sick_fish.mp3"}
                    )

    if not sound_playing:  # If no sound is playing
        if camera_index == PLANT_CAM_INDEX:
            sio.emit("stop_sound", {"sound_type": "plant"})
        if camera_index == FISH_CAM_INDEX:
            sio.emit("stop_sound", {"sound_type": "fish"})
        # elif camera_index == FISH_CAM_INDEX:
        #     sio.emit('stop_sound', {'sound_type': 'fish'})

    if camera_index == PLANT_CAM_INDEX:
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (220, 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(
            frame,
            f"Sick plants detected: {plant_cam_counter}",
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
            f"Average Size of Fish: {fish_cam_counter}",
            (10, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


# Function to generate video feed for a specific camera
def video_feed(camera: cv2.VideoCapture, camera_index: int, sio: socketio.SimpleClient):
    start_time = time.time()
    frame_count = 0

    while True:
        success, frame = camera.read()

        if not success:
            break

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

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        sio.emit("video_feed", {
            "frame": base64.b64encode(frame).decode("utf-8"),
            "camera_index": camera_index,
        })


with socketio.SimpleClient() as sio:
    print("Connecting to server")
    sio.connect("http://122.53.28.51:8000")
    print("Connected, starting feed")
    video_feed(plant_cam, PLANT_CAM_INDEX, sio)
    print("Feed stopped")