document.addEventListener("DOMContentLoaded", function() {
    function updateTime() {
        fetch('http://worldtimeapi.org/api/timezone/Asia/Shanghai')
            .then(response => response.json())
            .then(data => {

                const currentTime = new Date(data.utc_datetime);

                const timeOptions = { hour: '2-digit', minute: '2-digit', second: '2-digit', timeZone: 'Asia/Shanghai' };
                const dateOptions = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', timeZone: 'Asia/Shanghai' };
                const timeString = currentTime.toLocaleTimeString([], timeOptions);
                const dateString = currentTime.toLocaleDateString([], dateOptions);

                document.getElementById('current-time').textContent = timeString;
                document.getElementById('current-date').textContent = dateString;
            })
            .catch(error => {
                console.error('Error fetching time:', error);
            });
    }
    
    function playSound(soundType) {
        var audio = new Audio("/play_sound/" + soundType);
        audio.play();
    }
    
    updateTime(); 
    setInterval(updateTime, 1000);

    var socket = io();

    socket.on('play_sound', function(data) {
        playSound(data.sound_url);
    });
});
