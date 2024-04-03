var clientIdentifier = uuidv4();

document.addEventListener("DOMContentLoaded", function () {
    function updateTime() {
        fetch("http://worldtimeapi.org/api/timezone/Asia/Shanghai")
            .then((response) => response.json())
            .then((data) => {
                const currentTime = new Date(data.utc_datetime);

                const timeOptions = {
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                    timeZone: "Asia/Shanghai",
                };
                const dateOptions = {
                    weekday: "long",
                    year: "numeric",
                    month: "long",
                    day: "numeric",
                    timeZone: "Asia/Shanghai",
                };
                const timeString = currentTime.toLocaleTimeString(
                    [],
                    timeOptions
                );
                const dateString = currentTime.toLocaleDateString(
                    [],
                    dateOptions
                );

                document.getElementById("current-time").textContent =
                    timeString;
                document.getElementById("current-date").textContent =
                    dateString;
            })
            .catch((error) => {
                console.error("Error fetching time:", error);
            });
    }

    function playSound(soundType) {
        var audio = new Audio("/play_sound/" + soundType);
        audio.play();
    }

    updateTime();
    setInterval(updateTime, 1000);

    var socket = io();

    socket.on("video_feed", function (data) {
        const { frame, camera_index } = JSON.parse(data);
        const canvas = document.getElementById("cam-" + camera_index);
        const ctx = canvas.getContext("2d");
        const image = new Image();
        image.onload = function () {
            ctx.drawImage(image, 0, 0);
        };
        image.src = "data:image/png;base64," + frame;
    });

    socket.on("play_sound", function (data) {
        playSound(data.sound_url);
    });

    socket.emit("viewer_connected", {
        viewer_uuid: clientIdentifier,
    });
});

document.onvisibilitychange = () => {
    if (document.visibilityState === "hidden") {
        socket.emit("viewer_disconnecting", {
            viewer_uuid: clientIdentifier,
        });
    }
};

function uuidv4() {
    return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, (c) =>
        (
            c ^
            (crypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (c / 4)))
        ).toString(16)
    );
}