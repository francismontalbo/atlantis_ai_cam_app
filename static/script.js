document.addEventListener("DOMContentLoaded", function() {
    function updateTime() {
        const now = new Date();
        const utcOffset = 8;
        const localTime = new Date(now.getTime() + (utcOffset * 60 * 60 * 1000));
        const timeString = localTime.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'});
        const dateString = localTime.toDateString();
        document.getElementById('current-time').textContent = timeString;
        document.getElementById('current-date').textContent = dateString;
    }
    updateTime();
    setInterval(updateTime, 1000);
});
