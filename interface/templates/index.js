const btn = document.getElementById('btn');
const camera = document.getElementById('camera');
let running = false;


btn.addEventListener('click', async () => {
    if (!running) {
        camera.src = '/start_camera';
        btn.textContent = 'Stop Camera';
        running = true;
        camera.style.display = 'block';
    } else {
        await fetch('/stop_camera');
        camera.src = '';
        btn.textContent = 'Start Camera';
        running = false;
        camera.style.display = 'none';
    }
});

const alarm = new Audio('/static/alarm.wav');
const alarmBtn = document.getElementById('alarm');
alarmBtn.addEventListener('click', () => {
    if (alarmBtn.textContent === 'Activate Alarm') {
        alarm.currentTime = 0;
        alarm.play();
        alarmBtn.textContent = 'Stop Alarm';
    } else {
        alarm.pause();
        alarm.currentTime = 0;
        alarmBtn.textContent = 'Activate Alarm';
    }
})