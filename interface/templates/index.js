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

async function update_status() {
    const response = await fetch('/print_info');
    const data = await response.json();

    document.getElementById('name').textContent = data.name;
    document.getElementById('score').textContent = data.score.toFixed(2);
}

async function check_alarm() {
    const response = await fetch('/alarm_status');
    const data = await response.json();

    if (data.active) {
        alarm.currentTime = 0;
        alarm.play();
        alarmBtn.style.display = 'block';
    } else {
        alarm.pause();
        alarm.currentTime = 0;
        alarmBtn.style.display = 'none';
    }
}