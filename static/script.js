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
        const timeString = currentTime.toLocaleTimeString([], timeOptions);
        const dateString = currentTime.toLocaleDateString([], dateOptions);

        document.getElementById("current-time").textContent = timeString;
        document.getElementById("current-date").textContent = dateString;

        // Get timestamp
        const timestamp = currentTime.getTime();

        // Update plot with timestamp
        updatePlot(timestamp);
      })
      .catch((error) => {
        console.error("Error fetching time:", error);
      });
  }

  // Define initial empty data
  const data = [
    {
      x: [],
      y: [],
      mode: "lines",
      type: "scatter",
      line: { color: "#4CAF50" }, // Custom line color
    },
  ];

  // Define Layout
  const layout = {
    xaxis: { title: "Time", type: "date", tickformat: "%H:%M:%S" }, // Custom tick format
    yaxis: { title: "Sick Plant Count" },
    title: {
      text: "Sick Plants Detected Over Time",
      font: {
        size: 20,
        family: "Arial, sans-serif", // Custom font family
        color: "#333", // Custom title color
      },
    },
    plot_bgcolor: "#f0f0f0", // Custom plot background color
    paper_bgcolor: "#ffffff", // Custom paper background color
    margin: { t: 50, r: 50, b: 50, l: 50 }, // Custom margin
    hovermode: "closest", // Closest hover mode
    autosize: true, // Autosize enabled
  };

  // Display using Plotly
  Plotly.newPlot("myPlot", data, layout);

  // Function to update the plot with new data
  function updatePlot(timestamp) {
    Plotly.extendTraces(
      "myPlot",
      {
        x: [[new Date(timestamp)]],
        y: [[sick_plant_counter]],
      },
      [0]
    );
  }

  // Initialize time
  updateTime();
  setInterval(updateTime, 1000);

  // SocketIO connection to receive sick plant count data
  const socket = io();

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

  socket.emit("viewer_connected", {
    viewer_uuid: clientIdentifier,
  });

  socket.on("sick_plant_count", function (data) {
    updatePlot(Date.now(), data.count);
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
