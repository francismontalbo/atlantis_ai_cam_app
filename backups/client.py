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

    plants_counted = 0
    fish_counted = 0

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

            if label == "person":  # label == "sick_plant"
                plants_counted += 1

            if label == "chair":  # label in ["small_fish", "medium_fish", "big_fish"]
                fish_counted += 1

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
            f"Average Size of Fish: {fish_counted}",
            (10, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if plants_counted > plant_cam_counter:
        sio.emit(
            "play_sound", {"sound_url": "static/fx/sick_plant.mp3"}
        )
    
    if fish_counted > fish_cam_counter:
        sio.emit(
            "play_sound", {"sound_url": "static/fx/sick_fish.mp3"}
        )
    
    plant_cam_counter = plants_counted
    fish_cam_counter = fish_counted

    return frame


# Function to generate video feed for a specific camera
# def video_feed(camera: cv2.VideoCapture, camera_index: int, sio: socketio.SimpleClient):
#     start_time = time.time()
#     frame_count = 0

#     while True:
#         success, frame = camera.read()

#         if not success:
#             break

#         frame = detect_objects(frame, camera_index, sio)
#         frame_count += 1
#         elapsed_time = time.time() - start_time
#         fps = frame_count / elapsed_time

#         if SHOW_FPS:
#             height, width, _ = frame.shape
#             position = (width - 100, 30)
#             cv2.putText(
#                 frame,
#                 f"FPS: {fps:.2f}",
#                 position,
#                 cv2.FONT_HERSHEY_DUPLEX,
#                 0.4,
#                 (0, 0, 255),
#                 1,
#                 cv2.LINE_AA,
#             )

#         ret, buffer = cv2.imencode(".jpg", frame)
#         frame = buffer.tobytes()
#         sio.emit("video_feed", {
#             "frame": base64.b64encode(frame).decode("utf-8"),
#             "camera_index": camera_index,
#         })

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
        _, compressed_frame = cv2.imencode('.jpg', frame, encode_param)

        # Convert to bytes and send to server
        frame_bytes = compressed_frame.tobytes()
        sio.emit("video_feed", {
            "frame": base64.b64encode(frame_bytes).decode("utf-8"),
            "camera_index": camera_index,
        })


with socketio.SimpleClient() as sio:
    print("Connecting to server")
    sio.connect("http://122.53.28.51:8000")
    # sio.connect("http://127.0.0.1:8000")
    print("Connected, starting feed")
    video_feed(plant_cam, PLANT_CAM_INDEX, sio)
    print("Feed stopped")

# import base64
# import cv2
# import torchvision.transforms as transforms
# import torchvision
# import torch
# import time
# import pygame
# import socketio


# PLANT_CAM_INDEX = 0
# FISH_CAM_INDEX = 1
# FRAME_RATE = 60
# FRAME_WIDTH = 320
# FRAME_HEIGHT = 320
# SHOW_FPS = True

# pygame.mixer.init()
# sick_plant_sound = pygame.mixer.Sound("static/fx/sick_plant.mp3")
# sick_fish_sound = pygame.mixer.Sound("static/fx/sick_fish.mp3")

# plant_sound_interval = 10
# fish_sound_interval = 10

# plant_cam_counter = 0
# fish_cam_counter = 0

# last_played_time_plant = time.time()
# last_played_time_fish = time.time()


# def initialize_camera(
#     index, frame_rate=FRAME_RATE, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT
# ):
#     camera = cv2.VideoCapture(index)
#     camera.set(cv2.CAP_PROP_FPS, frame_rate)
#     camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
#     camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
#     return camera


# plant_cam = initialize_camera(PLANT_CAM_INDEX)
# fish_cam = initialize_camera(FISH_CAM_INDEX)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# weights = "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1"
# box_score_thresh = 0.9
# model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
#     pretrained=True, weights=weights, box_score_thresh=box_score_thresh
# )
# model = model.to(device)
# model.eval()


# def load_classes(file_path):
#     with open(file_path, "r") as file:
#         classes = [line.strip() for line in file.readlines() if line.strip()]
#     return classes


# classes_file = "classes.txt"
# CLASSES = load_classes(classes_file)


# # Function to perform object detection on frames
# def detect_objects(frame, camera_index: int, sio: socketio.SimpleClient):
#     global plant_cam_counter, fish_cam_counter, last_played_time_plant, last_played_time_fish

#     transform = transforms.Compose([transforms.ToTensor()])
#     image_tensor = transform(frame).unsqueeze(0).to(device)
#     with torch.no_grad():
#         prediction = model(image_tensor)

#     plants_counted = 0
#     fish_counted = 0

#     for idx in range(len(prediction[0]["boxes"])):
#         box = prediction[0]["boxes"][idx].cpu().numpy()
#         label = CLASSES[prediction[0]["labels"][idx].item()]
#         score = prediction[0]["scores"][idx].item()

#         if score > 0.5:
#             cv2.rectangle(
#                 frame,
#                 (int(box[0]), int(box[1])),
#                 (int(box[2]), int(box[3])),
#                 (0, 255, 0),
#                 2,
#             )
#             score_formatted = "{:.2f}".format(score * 100)
#             cv2.putText(
#                 frame,
#                 f"{label}: {score_formatted}%",
#                 (int(box[0]), int(box[1])),
#                 cv2.FONT_HERSHEY_DUPLEX,
#                 0.5,
#                 (255, 0, 0),
#                 1,
#             )

#             if label == "person":  # label == "sick_plant"
#                 plants_counted += 1

#             if label == "chair":  # label in ["small_fish", "medium_fish", "big_fish"]
#                 fish_counted += 1

#     if camera_index == PLANT_CAM_INDEX:
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (5, 5), (220, 35), (0, 0, 0), -1)
#         cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
#         cv2.putText(
#             frame,
#             f"Sick plants detected: {plants_counted}",
#             (10, 30),
#             cv2.FONT_HERSHEY_DUPLEX,
#             0.5,
#             (255, 255, 255),
#             1,
#             cv2.LINE_AA,
#         )
#     if camera_index == FISH_CAM_INDEX:
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (5, 5), (220, 35), (0, 0, 0), -1)
#         cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
#         cv2.putText(
#             frame,
#             f"Average Size of Fish: {fish_counted}",
#             (10, 30),
#             cv2.FONT_HERSHEY_DUPLEX,
#             0.5,
#             (255, 255, 255),
#             1,
#             cv2.LINE_AA,
#         )

#     if plants_counted > plant_cam_counter:
#         sio.emit(
#             "play_sound", {"sound_url": "static/fx/sick_plant.mp3"}
#         )
    
#     if fish_counted > fish_cam_counter:
#         sio.emit(
#             "play_sound", {"sound_url": "static/fx/sick_fish.mp3"}
#         )
    
#     plant_cam_counter = plants_counted
#     fish_cam_counter = fish_counted

#     return frame


# # Function to generate video feed for a specific camera
# def video_feed(camera: cv2.VideoCapture, camera_index: int, sio: socketio.SimpleClient):
#     start_time = time.time()
#     frame_count = 0

#     # Define the codec and create a VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     out = cv2.VideoWriter(f'camera_{camera_index}_output.avi', fourcc, FRAME_RATE, (FRAME_WIDTH, FRAME_HEIGHT))

#     while True:
#         success, frame = camera.read()

#         if not success:
#             break

#         # Perform object detection
#         frame = detect_objects(frame, camera_index, sio)
#         frame_count += 1
#         elapsed_time = time.time() - start_time
#         fps = frame_count / elapsed_time

#         if SHOW_FPS:
#             height, width, _ = frame.shape
#             position = (width - 100, 30)
#             cv2.putText(
#                 frame,
#                 f"FPS: {fps:.2f}",
#                 position,
#                 cv2.FONT_HERSHEY_DUPLEX,
#                 0.4,
#                 (0, 0, 255),
#                 1,
#                 cv2.LINE_AA,
#             )

#         # Write the frame to the output file
#         out.write(frame)

#         # Apply compression
#         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]  # adjust quality as needed
#         _, compressed_frame = cv2.imencode('.jpg', frame, encode_param)

#         # Convert to bytes and send to server
#         frame_bytes = compressed_frame.tobytes()
#         sio.emit("video_feed", {
#             "frame": base64.b64encode(frame_bytes).decode("utf-8"),
#             "camera_index": camera_index,
#         })

#     # Release the VideoWriter object
#     out.release()


# with socketio.SimpleClient() as sio:
#     print("Connecting to server")
#     sio.connect("http://122.53.28.51:8000")
#     # sio.connect("http://127.0.0.1:8000")
#     print("Connected, starting feed")
#     video_feed(plant_cam, PLANT_CAM_INDEX, sio)
#     print("Feed stopped")


# import base64
# import cv2
# import torchvision.transforms as transforms
# import torchvision.models.detection as detection
# import torch
# import time
# import pygame
# import socketio

# PLANT_CAM_INDEX = 0
# FISH_CAM_INDEX = 1
# FRAME_WIDTH = 320
# FRAME_HEIGHT = 320
# SHOW_FPS = True

# # Initialize Pygame mixer
# pygame.mixer.init()

# # Load sound effects
# sick_plant_sound = pygame.mixer.Sound("static/fx/sick_plant.mp3")
# sick_fish_sound = pygame.mixer.Sound("static/fx/sick_fish.mp3")

# # Initialize counters
# plant_cam_counter = 0
# fish_cam_counter = 0

# # Initialize camera
# def initialize_camera(index, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT):
#     camera = cv2.VideoCapture(index)
#     camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
#     camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
#     return camera

# # Initialize cameras
# plant_cam = initialize_camera(PLANT_CAM_INDEX)
# fish_cam = initialize_camera(FISH_CAM_INDEX)

# # Check for CUDA availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load pre-trained model
# model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)

# # Apply post-training quantization
# model = torch.quantization.quantize_dynamic(
#     model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
# )

# # Move model to appropriate device
# model = model.to(device)
# model.eval()

# # Load classes
# def load_classes(file_path):
#     with open(file_path, "r") as file:
#         classes = [line.strip() for line in file.readlines() if line.strip()]
#     return classes

# # Load class labels
# classes_file = "classes.txt"
# CLASSES = load_classes(classes_file)

# # Object detection function
# def detect_objects(frame, camera_index, sio):
#     global plant_cam_counter, fish_cam_counter

#     transform = transforms.Compose([transforms.ToTensor()])
#     image_tensor = transform(frame).unsqueeze(0).to(device)

#     with torch.no_grad():
#         prediction = model(image_tensor)

#     plants_counted = 0
#     fish_counted = 0

#     for idx in range(len(prediction[0]["boxes"])):
#         box = prediction[0]["boxes"][idx].cpu().numpy()
#         label = CLASSES[prediction[0]["labels"][idx].item()]
#         score = prediction[0]["scores"][idx].item()

#         if score > 0.5:
#             cv2.rectangle(
#                 frame,
#                 (int(box[0]), int(box[1])),
#                 (int(box[2]), int(box[3])),
#                 (0, 255, 0),
#                 2,
#             )
#             score_formatted = "{:.2f}".format(score * 100)
#             cv2.putText(
#                 frame,
#                 f"{label}: {score_formatted}%",
#                 (int(box[0]), int(box[1])),
#                 cv2.FONT_HERSHEY_DUPLEX,
#                 0.5,
#                 (255, 0, 0),
#                 1,
#             )

#             if label == "person":
#                 plants_counted += 1

#             if label == "chair":
#                 fish_counted += 1

#     overlay = frame.copy()
#     cv2.rectangle(overlay, (5, 5), (220, 35), (0, 0, 0), -1)
#     cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

#     if camera_index == PLANT_CAM_INDEX:
#         cv2.putText(
#             frame,
#             f"Sick plants detected: {plants_counted}",
#             (10, 30),
#             cv2.FONT_HERSHEY_DUPLEX,
#             0.5,
#             (255, 255, 255),
#             1,
#             cv2.LINE_AA,
#         )

#     if camera_index == FISH_CAM_INDEX:
#         cv2.putText(
#             frame,
#             f"Average Size of Fish: {fish_counted}",
#             (10, 30),
#             cv2.FONT_HERSHEY_DUPLEX,
#             0.5,
#             (255, 255, 255),
#             1,
#             cv2.LINE_AA,
#         )

#     if plants_counted > plant_cam_counter:
#         sio.emit(
#             "play_sound", {"sound_url": "static/fx/sick_plant.mp3"}
#         )

#     if fish_counted > fish_cam_counter:
#         sio.emit(
#             "play_sound", {"sound_url": "static/fx/sick_fish.mp3"}
#         )

#     plant_cam_counter = plants_counted
#     fish_cam_counter = fish_counted

#     return frame

# # Video feed function
# def video_feed(camera, camera_index, sio):
#     start_time = time.time()
#     frame_count = 0

#     while True:
#         success, frame = camera.read()

#         if not success:
#             break

#         frame = detect_objects(frame, camera_index, sio)
#         frame_count += 1
#         elapsed_time = time.time() - start_time
#         fps = frame_count / elapsed_time

#         if SHOW_FPS:
#             height, width, _ = frame.shape
#             position = (width - 100, 30)
#             cv2.putText(
#                 frame,
#                 f"FPS: {fps:.2f}",
#                 position,
#                 cv2.FONT_HERSHEY_DUPLEX,
#                 0.4,
#                 (0, 0, 255),
#                 1,
#                 cv2.LINE_AA,
#             )

#         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
#         _, compressed_frame = cv2.imencode('.jpeg', frame, encode_param)

#         frame_bytes = compressed_frame.tobytes()
#         sio.emit("video_feed", {
#             "frame": base64.b64encode(frame_bytes).decode("utf-8"),
#             "camera_index": camera_index,
#         })

# # Connect to server and start video feed
# with socketio.SimpleClient() as sio:
#     print("Connecting to server")
#     sio.connect("http://122.53.28.51:8000")
#     print("Connected, starting feed")
#     video_feed(plant_cam, PLANT_CAM_INDEX, sio)
#     print("Feed stopped")


# import base64
# import cv2
# import torchvision.transforms as transforms
# import torchvision.models.detection as detection
# import torch
# import time
# import pygame
# import socketio

# PLANT_CAM_INDEX = 0
# FISH_CAM_INDEX = 1
# FRAME_WIDTH = 320
# FRAME_HEIGHT = 320
# SHOW_FPS = True

# # Initialize Pygame mixer
# pygame.mixer.init()

# # Load sound effects
# sick_plant_sound = pygame.mixer.Sound("static/fx/sick_plant.mp3")
# sick_fish_sound = pygame.mixer.Sound("static/fx/sick_fish.mp3")

# # Initialize counters
# plant_cam_counter = 0
# fish_cam_counter = 0

# # Initialize camera
# def initialize_camera(index, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT):
#     camera = cv2.VideoCapture(index)
#     camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
#     camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
#     return camera

# # Initialize cameras
# plant_cam = initialize_camera(PLANT_CAM_INDEX)
# fish_cam = initialize_camera(FISH_CAM_INDEX)

# # Check for CUDA availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load pre-trained model
# model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)

# # Apply post-training quantization
# model = torch.quantization.quantize_dynamic(
#     model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
# )

# # Move model to appropriate device
# model = model.to(device)
# model.eval()

# # Load classes
# def load_classes(file_path):
#     with open(file_path, "r") as file:
#         classes = [line.strip() for line in file.readlines() if line.strip()]
#     return classes

# # Load class labels
# classes_file = "classes.txt"
# CLASSES = load_classes(classes_file)

# # Initialize an empty list to store sick plant detection times
# sick_plant_times = []

# # Object detection function
# def detect_objects(frame, camera_index, sio):
#     global plant_cam_counter, fish_cam_counter, sick_plant_times

#     transform = transforms.Compose([transforms.ToTensor()])
#     image_tensor = transform(frame).unsqueeze(0).to(device)

#     with torch.no_grad():
#         prediction = model(image_tensor)

#     plants_counted = 0
#     fish_counted = 0

#     for idx in range(len(prediction[0]["boxes"])):
#         box = prediction[0]["boxes"][idx].cpu().numpy()
#         label = CLASSES[prediction[0]["labels"][idx].item()]
#         score = prediction[0]["scores"][idx].item()

#         if score > 0.5:
#             cv2.rectangle(
#                 frame,
#                 (int(box[0]), int(box[1])),
#                 (int(box[2]), int(box[3])),
#                 (0, 255, 0),
#                 2,
#             )
#             score_formatted = "{:.2f}".format(score * 100)
#             cv2.putText(
#                 frame,
#                 f"{label}: {score_formatted}%",
#                 (int(box[0]), int(box[1])),
#                 cv2.FONT_HERSHEY_DUPLEX,
#                 0.5,
#                 (255, 0, 0),
#                 1,
#             )

#             if label == "person":
#                 plants_counted += 1
#                 # Record the time when a sick plant is detected
#                 current_time = time.time()  # Record the current time
#                 sick_plant_times.append(current_time)  # Add the time to the list

#             if label == "chair":
#                 fish_counted += 1

#     overlay = frame.copy()
#     cv2.rectangle(overlay, (5, 5), (220, 35), (0, 0, 0), -1)
#     cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

#     if camera_index == PLANT_CAM_INDEX:
#         cv2.putText(
#             frame,
#             f"Sick plants detected: {plants_counted}",
#             (10, 30),
#             cv2.FONT_HERSHEY_DUPLEX,
#             0.5,
#             (255, 255, 255),
#             1,
#             cv2.LINE_AA,
#         )

#     if camera_index == FISH_CAM_INDEX:
#         cv2.putText(
#             frame,
#             f"Average Size of Fish: {fish_counted}",
#             (10, 30),
#             cv2.FONT_HERSHEY_DUPLEX,
#             0.5,
#             (255, 255, 255),
#             1,
#             cv2.LINE_AA,
#         )

#     if plants_counted > plant_cam_counter:
#         sio.emit(
#             "play_sound", {"sound_url": "static/fx/sick_plant.mp3"}
#         )

#     if fish_counted > fish_cam_counter:
#         sio.emit(
#             "play_sound", {"sound_url": "static/fx/sick_fish.mp3"}
#         )

#     plant_cam_counter = plants_counted
#     fish_cam_counter = fish_counted

#     return frame

# # Video feed function
# def video_feed(camera, camera_index, sio):
#     start_time = time.time()
#     frame_count = 0

#     while True:
#         success, frame = camera.read()

#         if not success:
#             break

#         frame = detect_objects(frame, camera_index, sio)
#         frame_count += 1
#         elapsed_time = time.time() - start_time
#         fps = frame_count / elapsed_time

#         if SHOW_FPS:
#             height, width, _ = frame.shape
#             position = (width - 100, 30)
#             cv2.putText(
#                 frame,
#                 f"FPS: {fps:.2f}",
#                 position,
#                 cv2.FONT_HERSHEY_DUPLEX,
#                 0.4,
#                 (0, 0, 255),
#                 1,
#                 cv2.LINE_AA,
#             )

#         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
#         _, compressed_frame = cv2.imencode('.jpeg', frame, encode_param)

#         frame_bytes = compressed_frame.tobytes()
#         sio.emit("video_feed", {
#             "frame": base64.b64encode(frame_bytes).decode("utf-8"),
#             "camera_index": camera_index,
#         })

# # Connect to server and start video feed
# with socketio.SimpleClient() as sio:
#     print("Connecting to server")
#     # sio.connect("http://122.53.28.51:8000")
#     sio.connect("http://172.16.152.153:5000")
#     print("Connected, starting feed")
#     video_feed(plant_cam, PLANT_CAM_INDEX, sio)
#     print("Feed stopped")


# Your code with adjustments for localhost

# import base64
# import cv2
# import torchvision.transforms as transforms
# import torchvision.models.detection as detection
# import torch
# import time
# import pygame
# import socketio

# PLANT_CAM_INDEX = 0
# FISH_CAM_INDEX = 1
# FRAME_WIDTH = 320
# FRAME_HEIGHT = 320
# SHOW_FPS = True

# # Initialize Pygame mixer
# pygame.mixer.init()

# # Load sound effects
# sick_plant_sound = pygame.mixer.Sound("static/fx/sick_plant.mp3")
# sick_fish_sound = pygame.mixer.Sound("static/fx/sick_fish.mp3")

# # Initialize counters
# plant_cam_counter = 0
# fish_cam_counter = 0

# # Initialize camera
# def initialize_camera(index=0, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT):
#     camera = cv2.VideoCapture(index)
#     camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
#     camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
#     return camera

# # Initialize cameras
# plant_cam = initialize_camera(PLANT_CAM_INDEX)
# fish_cam = initialize_camera(FISH_CAM_INDEX)

# # Check for CUDA availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load pre-trained model
# model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)

# # Apply post-training quantization
# model = torch.quantization.quantize_dynamic(
#     model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
# )

# # Move model to appropriate device
# model = model.to(device)
# model.eval()

# # Load classes
# def load_classes(file_path):
#     with open(file_path, "r") as file:
#         classes = [line.strip() for line in file.readlines() if line.strip()]
#     return classes

# # Load class labels
# classes_file = "classes.txt"
# CLASSES = load_classes(classes_file)

# # Initialize an empty list to store sick plant detection times
# sick_plant_times = []

# # Object detection function
# def detect_objects(frame, camera_index, sio):
#     global plant_cam_counter, fish_cam_counter, sick_plant_times

#     transform = transforms.Compose([transforms.ToTensor()])
#     image_tensor = transform(frame).unsqueeze(0).to(device)

#     with torch.no_grad():
#         prediction = model(image_tensor)

#     plants_counted = 0
#     fish_counted = 0

#     for idx in range(len(prediction[0]["boxes"])):
#         box = prediction[0]["boxes"][idx].cpu().numpy()
#         label = CLASSES[prediction[0]["labels"][idx].item()]
#         score = prediction[0]["scores"][idx].item()

#         if score > 0.5:
#             cv2.rectangle(
#                 frame,
#                 (int(box[0]), int(box[1])),
#                 (int(box[2]), int(box[3])),
#                 (0, 255, 0),
#                 2,
#             )
#             score_formatted = "{:.2f}".format(score * 100)
#             cv2.putText(
#                 frame,
#                 f"{label}: {score_formatted}%",
#                 (int(box[0]), int(box[1])),
#                 cv2.FONT_HERSHEY_DUPLEX,
#                 0.5,
#                 (255, 0, 0),
#                 1,
#             )

#             if label == "person":
#                 plants_counted += 1
#                 # Record the time when a sick plant is detected
#                 current_time = time.time()  # Record the current time
#                 sick_plant_times.append(current_time)  # Add the time to the list

#             if label == "chair":
#                 fish_counted += 1

#     overlay = frame.copy()
#     cv2.rectangle(overlay, (5, 5), (220, 35), (0, 0, 0), -1)
#     cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

#     if camera_index == PLANT_CAM_INDEX:
#         cv2.putText(
#             frame,
#             f"Sick plants detected: {plants_counted}",
#             (10, 30),
#             cv2.FONT_HERSHEY_DUPLEX,
#             0.5,
#             (255, 255, 255),
#             1,
#             cv2.LINE_AA,
#         )

#     if camera_index == FISH_CAM_INDEX:
#         cv2.putText(
#             frame,
#             f"Average Size of Fish: {fish_counted}",
#             (10, 30),
#             cv2.FONT_HERSHEY_DUPLEX,
#             0.5,
#             (255, 255, 255),
#             1,
#             cv2.LINE_AA,
#         )

#     if plants_counted > plant_cam_counter:
#         sio.emit(
#             "play_sound", {"sound_url": "static/fx/sick_plant.mp3"}
#         )

#     if fish_counted > fish_cam_counter:
#         sio.emit(
#             "play_sound", {"sound_url": "static/fx/sick_fish.mp3"}
#         )

#     plant_cam_counter = plants_counted
#     fish_cam_counter = fish_counted

#     return frame

# # Video feed function
# def video_feed(camera, camera_index, sio):
#     start_time = time.time()
#     frame_count = 0

#     while True:
#         success, frame = camera.read()

#         if not success:
#             break

#         frame = detect_objects(frame, camera_index, sio)
#         frame_count += 1
#         elapsed_time = time.time() - start_time
#         fps = frame_count / elapsed_time

#         if SHOW_FPS:
#             height, width, _ = frame.shape
#             position = (width - 100, 30)
#             cv2.putText(
#                 frame,
#                 f"FPS: {fps:.2f}",
#                 position,
#                 cv2.FONT_HERSHEY_DUPLEX,
#                 0.4,
#                 (0, 0, 255),
#                 1,
#                 cv2.LINE_AA,
#             )
        
#         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
#         _, compressed_frame = cv2.imencode('.jpeg', frame, encode_param)

#         frame_bytes = compressed_frame.tobytes()
#         sio.emit("video_feed", {
#             "frame": base64.b64encode(frame_bytes).decode("utf-8"),
#             "camera_index": camera_index,
#         })


# # Connect to server and start video feed
# with socketio.SimpleClient() as sio:
#     print("Connecting to server")
#     sio.connect("http://localhost:5000")
#     print("Connected, starting feed")
#     video_feed(plant_cam, PLANT_CAM_INDEX, sio)
#     print("Feed stopped")

