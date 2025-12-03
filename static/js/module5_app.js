// Main application logic for Module 5-6 Object Tracking

let video = null;
let canvas = null;
let ctx = null;
let stream = null;
let tracker = null;
let isRunning = false;
let isSelectingRegion = false;
let selectionStart = null;
let selectionRect = null;
let frameCount = 0;
let fps = 0;
let lastFrameTime = 0;

// Initialize when OpenCV is ready
function onOpenCvReady() {
    console.log('OpenCV.js is ready');
    tracker = new ObjectTracker();
    initializeUI();
}

// Check if OpenCV is already loaded
if (typeof cv !== 'undefined') {
    onOpenCvReady();
} else {
    cv['onRuntimeInitialized'] = onOpenCvReady;
}

function initializeUI() {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d', { willReadFrequently: true });

    const trackingMode = document.querySelectorAll('input[name="trackingMode"]');
    trackingMode.forEach(radio => {
        radio.addEventListener('change', onModeChange);
    });

    // Canvas mouse handlers for region selection
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    
    // NPZ file selection handler
    const sam2FileInput = document.getElementById('sam2FileHidden');
    if (sam2FileInput) {
        sam2FileInput.addEventListener('change', function() {
            const fileName = this.files[0] ? this.files[0].name : 'No file selected';
            document.getElementById('sam2FileName').textContent = fileName;
            document.getElementById('loadSam2Btn').disabled = !this.files[0];
        });
    }
}

function onModeChange() {
    const mode = document.querySelector('input[name="trackingMode"]:checked').value;
    tracker.setMode(mode);

    const sam2Controls = document.getElementById('sam2Controls');
    const selectRegionBtn = document.getElementById('selectRegionBtn');

    if (mode === 'sam2') {
        sam2Controls.style.display = 'block';
        selectRegionBtn.disabled = true;
    } else {
        sam2Controls.style.display = 'none';
        if (isRunning) {
            selectRegionBtn.disabled = (mode !== 'markerless');
        }
    }

    updateStatus(`Mode changed to: ${mode}`);
}

async function startCamera() {
    const progressContainer = document.getElementById('progressContainer');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');

    progressContainer.classList.add('active');
    progressFill.style.width = '20%';
    progressFill.textContent = '20%';
    progressText.textContent = 'Starting camera...';

    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30 }
            } 
        });

        video.srcObject = stream;

        progressFill.style.width = '50%';
        progressFill.textContent = '50%';
        progressText.textContent = 'Loading...';

        await video.play();

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        isRunning = true;

        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;

        const mode = document.querySelector('input[name="trackingMode"]:checked').value;
        if (mode === 'markerless') {
            document.getElementById('selectRegionBtn').disabled = false;
        }

        progressFill.style.width = '100%';
        progressFill.textContent = '100%';
        progressText.textContent = 'Ready! Tracking...';

        setTimeout(() => progressContainer.classList.remove('active'), 800);

        processFrame();

    } catch (err) {
        console.error('Camera error:', err);
        alert('Camera error: ' + err.message + '\n\nEnsure camera permissions are granted.');
        progressContainer.classList.remove('active');
    }
}

function stopCamera() {
    isRunning = false;

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    video.srcObject = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('selectRegionBtn').disabled = true;

    frameCount = 0;
    fps = 0;
    updateStats();
    updateStatus('Camera stopped');
}

function toggleRegionSelection() {
    isSelectingRegion = !isSelectingRegion;
    const btn = document.getElementById('selectRegionBtn');

    if (isSelectingRegion) {
        btn.textContent = 'Cancel Selection';
        btn.classList.remove('btn-secondary');
        btn.classList.add('btn-danger');
        updateStatus('Click and drag on video to select region');
    } else {
        btn.textContent = 'Select Region';
        btn.classList.remove('btn-danger');
        btn.classList.add('btn-secondary');
        selectionRect = null;
        updateStatus('Selection cancelled');
    }
}

function onMouseDown(e) {
    if (!isSelectingRegion || !isRunning) return;

    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);

    selectionStart = { x, y };
    selectionRect = null;
}

function onMouseMove(e) {
    if (!isSelectingRegion || !selectionStart || !isRunning) return;

    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);

    selectionRect = {
        x: Math.min(selectionStart.x, x),
        y: Math.min(selectionStart.y, y),
        width: Math.abs(x - selectionStart.x),
        height: Math.abs(y - selectionStart.y)
    };
}

function onMouseUp(e) {
    if (!isSelectingRegion || !selectionStart || !isRunning) return;

    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);

    selectionRect = {
        x: Math.min(selectionStart.x, x),
        y: Math.min(selectionStart.y, y),
        width: Math.abs(x - selectionStart.x),
        height: Math.abs(y - selectionStart.y)
    };

    if (selectionRect.width > 10 && selectionRect.height > 10) {
        captureTemplate(selectionRect);
        isSelectingRegion = false;
        const btn = document.getElementById('selectRegionBtn');
        btn.textContent = 'Select Region';
        btn.classList.remove('btn-danger');
        btn.classList.add('btn-secondary');
        updateStatus('Region selected. Tracking started.');
    } else {
        selectionRect = null;
        updateStatus('Selection too small. Try again.');
    }

    selectionStart = null;
}

function captureTemplate(rect) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0);

    const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    const src = cv.matFromImageData(imageData);

    tracker.setTemplate(rect, src);

    src.delete();
}

async function loadSAM2File() {
    const fileInput = document.getElementById('sam2FileHidden');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select an NPZ file first');
        return;
    }

    updateStatus('Loading SAM2 file...');
    document.getElementById('loadSam2Btn').disabled = true;
    document.getElementById('loadSam2Btn').textContent = 'Loading...';
    
    const reader = new FileReader();
    reader.onload = async function(e) {
        try {
            await tracker.loadSAM2Data(e.target.result);
            const maskCount = tracker.sam2Masks ? tracker.sam2Masks.length : 0;
            updateStatus(`SAM2 file loaded: ${maskCount} mask(s) ready`);
            document.getElementById('loadSam2Btn').textContent = '✓ Loaded';
        } catch (err) {
            console.error('Error loading SAM2 file:', err);
            updateStatus('Error loading SAM2 file: ' + err.message);
            document.getElementById('loadSam2Btn').disabled = false;
            document.getElementById('loadSam2Btn').textContent = 'Load Segmentation';
        }
    };
    reader.onerror = function() {
        updateStatus('Error reading file');
        document.getElementById('loadSam2Btn').disabled = false;
        document.getElementById('loadSam2Btn').textContent = 'Load Segmentation';
    };
    reader.readAsArrayBuffer(file);
}

function processFrame() {
    if (!isRunning) return;

    try {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const src = cv.matFromImageData(imageData);
        const dst = src.clone();

        let tracked = false;
        let objectCount = 0;

        if (!isSelectingRegion || tracker.template) {
            const result = tracker.processFrame(src, dst);
            tracked = result.tracked;
            objectCount = result.objectCount;
        }

        cv.imshow(canvas, dst);

        if (isSelectingRegion && selectionRect) {
            ctx.strokeStyle = '#FFD700';
            ctx.fillStyle = 'rgba(255, 215, 0, 0.2)';
            ctx.lineWidth = 3;
            ctx.setLineDash([5, 5]);
            ctx.fillRect(
                selectionRect.x,
                selectionRect.y,
                selectionRect.width,
                selectionRect.height
            );
            ctx.strokeRect(
                selectionRect.x,
                selectionRect.y,
                selectionRect.width,
                selectionRect.height
            );
            ctx.setLineDash([]);
        }

        src.delete();
        dst.delete();

        frameCount++;
        updateStats(objectCount);
        updateFPS();

        if (isSelectingRegion) {
            if (selectionRect) {
                updateStatus('Drag to adjust selection, release to confirm');
            } else {
                updateStatus('Click and drag on video to select region');
            }
        } else if (tracked) {
            updateStatus('Tracking: Object detected');
        } else if (tracker.template) {
            updateStatus('Tracking: Searching...');
        } else {
            updateStatus('Ready');
        }
    } catch (err) {
        console.error('Processing error:', err);
    }

    requestAnimationFrame(processFrame);
}

function updateFPS() {
    const now = performance.now();
    if (lastFrameTime > 0) {
        const delta = now - lastFrameTime;
        fps = Math.round(1000 / delta);
        document.getElementById('fpsCounter').textContent = `FPS: ${fps}`;
        document.getElementById('fpsDisplay').textContent = fps;
    }
    lastFrameTime = now;
}

function updateStats(objectCount = 0) {
    document.getElementById('frameCount').textContent = frameCount;
    document.getElementById('objectCount').textContent = objectCount;
}

function updateStatus(message) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = `Status: ${message}`;
}

function cleanupResources() {
    isRunning = false;

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }

    if (video.srcObject) {
        video.srcObject = null;
    }

    frameCount = 0;
}

window.addEventListener('beforeunload', cleanupResources);
window.addEventListener('pagehide', cleanupResources);
