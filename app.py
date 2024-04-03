import json
from flask import Flask, render_template, Response, send_file
from flask_socketio import SocketIO


app = Flask(__name__)
socketio = SocketIO(app)


@app.route("/play_sound/<sound_type>")
def play_sound(sound_type):
    if sound_type == "plant":
        sound_file = "static/fx/sick_plant.mp3"
    elif sound_type == "fish":
        sound_file = "static/fx/sick_fish.mp3"
    return send_file(sound_file)


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("video_feed")
def emit_feed(data):
    socketio.emit("video_feed", json.dumps(data))
    
@socketio.on('viewer_connected')
def broadcast_viewer_connected(details):
    socketio.emit("viewer_connected", details)
    
@socketio.on('viewer_disconnecting')
def broadcast_viewer_disconnecting(details):
    socketio.emit("viewer_disconnecting", details)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8000, debug=True)
