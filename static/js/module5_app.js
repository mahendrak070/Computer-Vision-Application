// Module 5-6: Object Tracking Application

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
    updateStatus('Ready - Select a tracking mode and start camera');
}

// Check if OpenCV is already loaded
if (typeof cv !== 'undefined' && cv.Mat) {
    onOpenCvReady();
} else if (typeof cv !== 'undefined') {
    cv['onRuntimeInitialized'] = onOpenCvReady;
}

function initializeUI() {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d', { willReadFrequently: true });

    // Mode change handlers
    document.querySelectorAll('input[name="trackingMode"]').forEach(radio => {
        radio.addEventListener('change', onModeChange);
    });

    // Canvas mouse handlers
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    
    // Touch support
    canvas.addEventListener('touchstart', onTouchStart, { passive: false });
    canvas.addEventListener('touchmove', onTouchMove, { passive: false });
    canvas.addEventListener('touchend', onTouchEnd, { passive: false });
    
    // NPZ file handler
    const sam2Input = document.getElementById('sam2FileHidden');
    if (sam2Input) {
        sam2Input.addEventListener('change', function() {
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

    sam2Controls.style.display = (mode === 'sam2') ? 'block' : 'none';
    
    if (mode === 'markerless') {
        selectRegionBtn.disabled = !isRunning;
        if (isRunning) {
            updateStatus('Click "Select Region" to choose an object to track');
        }
    } else {
        selectRegionBtn.disabled = true;
        isSelectingRegion = false;
        selectionRect = null;
    }

    updateStatus(`Mode: ${mode}`);
}

async function startCamera() {
    const progressContainer = document.getElementById('progressContainer');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');

    progressContainer.classList.add('active');
    progressFill.style.width = '30%';
    progressText.textContent = 'Requesting camera...';

    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30 }
            } 
        });

        video.srcObject = stream;
        progressFill.style.width = '60%';
        progressText.textContent = 'Starting video...';

        await video.play();

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        isRunning = true;

        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;

        const mode = document.querySelector('input[name="trackingMode"]:checked').value;
        document.getElementById('selectRegionBtn').disabled = (mode !== 'markerless');

        progressFill.style.width = '100%';
        progressText.textContent = 'Camera ready!';
        setTimeout(() => progressContainer.classList.remove('active'), 500);

        if (mode === 'markerless') {
            updateStatus('Click "Select Region" to choose an object');
        } else if (mode === 'marker') {
            updateStatus('Show a square marker to the camera');
        } else {
            updateStatus('Load an NPZ file for SAM2 tracking');
        }

        processFrame();

    } catch (err) {
        console.error('Camera error:', err);
        alert('Camera error: ' + err.message);
        progressContainer.classList.remove('active');
    }
}

function stopCamera() {
    isRunning = false;
    isSelectingRegion = false;
    selectionRect = null;

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    video.srcObject = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('selectRegionBtn').disabled = true;

    const btn = document.getElementById('selectRegionBtn');
    btn.textContent = 'Select Region';
    btn.classList.remove('btn-danger');
    btn.classList.add('btn-secondary');

    frameCount = 0;
    fps = 0;
    updateStats();
    updateStatus('Camera stopped');
}

function toggleRegionSelection() {
    isSelectingRegion = !isSelectingRegion;
    const btn = document.getElementById('selectRegionBtn');

    if (isSelectingRegion) {
        btn.textContent = 'Cancel';
        btn.classList.remove('btn-secondary');
        btn.classList.add('btn-danger');
        tracker.reset(); // Clear any existing template
        updateStatus('CLICK AND DRAG on video to select object');
    } else {
        btn.textContent = 'Select Region';
        btn.classList.remove('btn-danger');
        btn.classList.add('btn-secondary');
        selectionRect = null;
        selectionStart = null;
        updateStatus('Selection cancelled');
    }
}

function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    let clientX, clientY;
    if (e.touches) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
    } else {
        clientX = e.clientX;
        clientY = e.clientY;
    }
    
    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY
    };
}

function onMouseDown(e) {
    if (!isSelectingRegion || !isRunning) return;
    e.preventDefault();
    
    const coords = getCanvasCoords(e);
    selectionStart = coords;
    selectionRect = null;
    console.log('Selection started at:', coords);
}

function onMouseMove(e) {
    if (!isSelectingRegion || !selectionStart || !isRunning) return;
    e.preventDefault();
    
    const coords = getCanvasCoords(e);
    selectionRect = {
        x: Math.min(selectionStart.x, coords.x),
        y: Math.min(selectionStart.y, coords.y),
        width: Math.abs(coords.x - selectionStart.x),
        height: Math.abs(coords.y - selectionStart.y)
    };
}

function onMouseUp(e) {
    if (!isSelectingRegion || !selectionStart || !isRunning) return;
    e.preventDefault();
    
    const coords = getCanvasCoords(e);
    selectionRect = {
        x: Math.min(selectionStart.x, coords.x),
        y: Math.min(selectionStart.y, coords.y),
        width: Math.abs(coords.x - selectionStart.x),
        height: Math.abs(coords.y - selectionStart.y)
    };
    
    console.log('Selection ended:', selectionRect);

    if (selectionRect.width > 30 && selectionRect.height > 30) {
        captureTemplate(selectionRect);
        finishSelection(true);
    } else {
        updateStatus('Selection too small - drag a larger area');
        selectionRect = null;
    }
    
    selectionStart = null;
}

// Touch handlers
function onTouchStart(e) {
    onMouseDown(e);
}

function onTouchMove(e) {
    onMouseMove(e);
}

function onTouchEnd(e) {
    if (!isSelectingRegion || !selectionStart || !isRunning) return;
    e.preventDefault();
    
    if (selectionRect && selectionRect.width > 30 && selectionRect.height > 30) {
        captureTemplate(selectionRect);
        finishSelection(true);
    } else {
        updateStatus('Selection too small');
        selectionRect = null;
    }
    
    selectionStart = null;
}

function finishSelection(success) {
    isSelectingRegion = false;
    selectionRect = null;
    
    const btn = document.getElementById('selectRegionBtn');
    btn.textContent = 'Select Region';
    btn.classList.remove('btn-danger');
    btn.classList.add('btn-secondary');
    
    if (success) {
        updateStatus('Object selected - now tracking');
    }
}

function captureTemplate(rect) {
    try {
        // Create temp canvas to capture current frame
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(video, 0, 0);

        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const src = cv.matFromImageData(imageData);

        console.log('Capturing template from rect:', rect);
        const success = tracker.setTemplate(rect, src);
        src.delete();
        
        if (success) {
            console.log('Template captured successfully');
            updateStatus('Tracking object...');
        } else {
            console.warn('Failed to capture template');
            updateStatus('Failed - try selecting a different area');
        }
        
        return success;
    } catch (e) {
        console.error('captureTemplate error:', e);
        updateStatus('Error capturing template');
        return false;
    }
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
            updateStatus(`Loaded ${maskCount} mask(s)`);
            document.getElementById('loadSam2Btn').textContent = '✓ Loaded';
        } catch (err) {
            console.error('SAM2 load error:', err);
            updateStatus('Error: ' + err.message);
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
        // Draw video to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Get frame as OpenCV mat
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const src = cv.matFromImageData(imageData);
        const dst = src.clone();

        let tracked = false;
        let objectCount = 0;

        // Only run tracking if NOT selecting
        if (!isSelectingRegion) {
            const result = tracker.processFrame(src, dst);
            tracked = result.tracked;
            objectCount = result.objectCount;
        }

        // Display result
        cv.imshow(canvas, dst);

        // Draw selection rectangle if selecting
        if (isSelectingRegion && selectionRect) {
            ctx.strokeStyle = '#00FF00';
            ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
            ctx.lineWidth = 3;
            ctx.fillRect(selectionRect.x, selectionRect.y, selectionRect.width, selectionRect.height);
            ctx.strokeRect(selectionRect.x, selectionRect.y, selectionRect.width, selectionRect.height);
            
            // Draw size indicator
            ctx.fillStyle = '#00FF00';
            ctx.font = '14px Arial';
            ctx.fillText(`${Math.round(selectionRect.width)} x ${Math.round(selectionRect.height)}`, 
                         selectionRect.x + 5, selectionRect.y - 5);
        }

        // Cleanup
        src.delete();
        dst.delete();

        // Update stats
        frameCount++;
        updateStats(objectCount);
        updateFPS();

        // Update status based on state
        if (!isSelectingRegion && tracker.trackingMode === 'markerless') {
            if (tracked) {
                // Status updated by tracking
            } else if (tracker.template) {
                updateStatus('Searching for object...');
            }
        }

    } catch (err) {
        console.error('Frame processing error:', err);
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
    document.getElementById('status').textContent = `Status: ${message}`;
}

// Cleanup
window.addEventListener('beforeunload', () => {
    isRunning = false;
    if (stream) stream.getTracks().forEach(track => track.stop());
});
