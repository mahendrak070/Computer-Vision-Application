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
    console.log('OpenCV.js ready');
    tracker = new ObjectTracker();
    initializeUI();
    updateStatus('Ready - Click "Start Camera"');
}

// Check if OpenCV is already loaded
if (typeof cv !== 'undefined' && cv.Mat) {
    onOpenCvReady();
} else {
    if (typeof cv !== 'undefined') {
        cv['onRuntimeInitialized'] = onOpenCvReady;
    } else {
        console.error('OpenCV.js not found');
    }
}

function initializeUI() {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    // Mode change handlers
    const trackingModes = document.querySelectorAll('input[name="trackingMode"]');
    trackingModes.forEach(radio => {
        radio.addEventListener('change', onModeChange);
    });
    
    // Canvas mouse handlers
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    
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
    if (tracker) tracker.setMode(mode);
    
    const sam2Controls = document.getElementById('sam2Controls');
    const selectRegionBtn = document.getElementById('selectRegionBtn');
    
    if (mode === 'sam2') {
        sam2Controls.style.display = 'block';
        selectRegionBtn.disabled = true;
    } else {
        sam2Controls.style.display = 'none';
        selectRegionBtn.disabled = !isRunning || mode !== 'markerless';
    }
    
    updateStatus(`Mode: ${mode}`);
}

async function startCamera() {
    const progressContainer = document.getElementById('progressContainer');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    
    progressContainer.classList.add('active');
    progressFill.style.width = '20%';
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
        
        updateStatus('Camera running');
        processFrame();
        
    } catch (err) {
        console.error('Camera error:', err);
        alert('Camera error: ' + err.message);
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
    updateStatus('Stopped');
}

function toggleRegionSelection() {
    isSelectingRegion = !isSelectingRegion;
    const btn = document.getElementById('selectRegionBtn');
    
    if (isSelectingRegion) {
        btn.textContent = 'Cancel';
        btn.classList.add('btn-danger');
        btn.classList.remove('btn-secondary');
        updateStatus('Click and drag to select object');
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
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    selectionStart = {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
    selectionRect = null;
}

function onMouseMove(e) {
    if (!isSelectingRegion || !selectionStart || !isRunning) return;
    
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    
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
    
    if (selectionRect && selectionRect.width > 20 && selectionRect.height > 20) {
        captureTemplate(selectionRect);
        
        isSelectingRegion = false;
        const btn = document.getElementById('selectRegionBtn');
        btn.textContent = 'Select Region';
        btn.classList.remove('btn-danger');
        btn.classList.add('btn-secondary');
        
        updateStatus('Template captured - Tracking started');
    } else {
        updateStatus('Selection too small, try again');
    }
    
    selectionStart = null;
}

function captureTemplate(rect) {
    try {
        // Draw current frame
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(video, 0, 0);
        
        // Get image data and create OpenCV mat
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const src = cv.matFromImageData(imageData);
        
        // Set template in tracker
        tracker.setTemplate(rect, src);
        
        src.delete();
        selectionRect = null;
    } catch (e) {
        console.error('Template capture error:', e);
        updateStatus('Error capturing template');
    }
}

async function loadSAM2File() {
    const fileInput = document.getElementById('sam2FileHidden');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select an NPZ file');
        return;
    }
    
    updateStatus('Loading SAM2 data...');
    document.getElementById('loadSam2Btn').disabled = true;
    document.getElementById('loadSam2Btn').textContent = 'Loading...';
    
    try {
        const arrayBuffer = await file.arrayBuffer();
        await tracker.loadSAM2Data(arrayBuffer);
        
        const maskCount = tracker.sam2Masks ? tracker.sam2Masks.length : 0;
        updateStatus(`Loaded ${maskCount} mask(s)`);
        document.getElementById('loadSam2Btn').textContent = '✓ Loaded';
    } catch (err) {
        console.error('SAM2 load error:', err);
        updateStatus('Error: ' + err.message);
        document.getElementById('loadSam2Btn').disabled = false;
        document.getElementById('loadSam2Btn').textContent = 'Load Segmentation';
    }
}

function processFrame() {
    if (!isRunning) return;
    
    try {
        // Draw video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Get frame as OpenCV mat
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const src = cv.matFromImageData(imageData);
        const dst = src.clone();
        
        // Process with tracker
        let tracked = false;
        let objectCount = 0;
        
        if (!isSelectingRegion) {
            const result = tracker.processFrame(src, dst);
            tracked = result.tracked;
            objectCount = result.objectCount;
        }
        
        // Display result
        cv.imshow(canvas, dst);
        
        // Draw selection rectangle if selecting
        if (isSelectingRegion && selectionRect) {
            ctx.strokeStyle = '#FFD700';
            ctx.fillStyle = 'rgba(255, 215, 0, 0.2)';
            ctx.lineWidth = 3;
            ctx.setLineDash([8, 4]);
            ctx.fillRect(selectionRect.x, selectionRect.y, selectionRect.width, selectionRect.height);
            ctx.strokeRect(selectionRect.x, selectionRect.y, selectionRect.width, selectionRect.height);
            ctx.setLineDash([]);
        }
        
        // Cleanup
        src.delete();
        dst.delete();
        
        // Update stats
        frameCount++;
        updateStats(objectCount);
        updateFPS();
        
        // Update status
        if (!isSelectingRegion) {
            if (tracked) {
                updateStatus(`Tracking: ${objectCount} object(s) detected`);
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

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    isRunning = false;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});
