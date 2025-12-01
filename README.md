# Computer Vision Web Application

**Live Demo:** [https://computervision.up.railway.app/](https://computervision.up.railway.app/)

**Authors:** Mahendra Krishna Koneru and Sai Leenath Jampala

A production-ready web-based computer vision platform featuring real-time tracking, image processing, SIFT stitching, dimension estimation, and advanced CV algorithms.

---

## 🚀 Quick Access

**Live Application:** [https://computervision.up.railway.app/](https://computervision.up.railway.app/)

**Developer Login:** [https://computervision.up.railway.app/dev-login](https://computervision.up.railway.app/dev-login)

---

## ✨ Features

### Computer Vision Modules

1. **Module 1: Dimension Estimation**
   - Real-world dimension measurement using perspective projection
   - Camera calibration integration (MTX & DIST)
   - Multi-point selection (up to 4 points)
   - Default: 5 inches known distance, 5238px focal length

2. **Module 2: Template Matching & Fourier Restoration**
   - Real-time template matching with bounding boxes
   - Gaussian blur and FFT-based restoration
   - Non-maximum suppression (NMS)
   - Progress tracking

3. **Module 3: Feature Detection & Analysis**
   - Gradient computation (magnitude & angle visualization)
   - Laplacian of Gaussian (LoG) edge detection
   - Custom edge detection with NMS & hysteresis thresholding
   - Harris corner detection with heatmap visualization
   - Boundary detection with intelligent object scoring
   - ArUco marker-based segmentation

4. **Module 4: Image Stitching**
   - Panorama creation from multiple images
   - FLANN-based feature matching
   - Homography estimation with RANSAC
   - Weighted blending

5. **Module 5-6: Object Tracking**
   - Client-side tracking using OpenCV.js
   - Marker-based tracking (ArUco, QR codes)
   - Marker-less tracking (template matching, optical flow)
   - SAM2 segmentation integration with NPZ file support
   - Real-time FPS and object count display

6. **Module 7: Pose & Hand Tracking**
   - Mediapipe integration for real-time tracking
   - Full body pose estimation with 33 landmarks
   - Body position detection (Standing, Sitting, Crouching, Lying Down, Leaning)
   - Hand tracking with gesture recognition (Left/Right hand detection)
   - Visibility quality metrics and landmark counts
   - CSV data export for pose and hand coordinates

---

## 🌐 Deployment

**Platform:** Railway  
**URL:** [https://computervision.up.railway.app/](https://computervision.up.railway.app/)  
**Status:** ✅ Live & Production Ready

### Auto-Deployment

Every push to `main` branch automatically deploys to Railway:

```bash
git add .
git commit -m "Your changes"
git push origin main
```

Railway will rebuild and deploy in ~3-5 minutes.

---

## 💻 Local Development

### Prerequisites

- Python 3.12
- Git

### Setup

1. **Clone repository:**
```bash
git clone https://github.com/mahendrak070/Computer-Vision-Application.git
cd Computer-Vision-Application
```

2. **Create virtual environment:**
```bash
python3.12 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run application:**
```bash
python app.py
```

5. **Access locally:**
```
http://localhost:5001
```

---

## 🔐 Authentication

**Production (Railway):** Face authentication is disabled to prevent build complications.

**Access Methods:**

1. **Developer Login** (Recommended)
   - URL: [https://computervision.up.railway.app/dev-login](https://computervision.up.railway.app/dev-login)
   - One-click access to all modules
   - No registration required

2. **Face Authentication** (Local Only)
   - Works on local development server
   - Requires `dlib` and `face-recognition` libraries
   - Install locally: `pip install face-recognition dlib`

---

## 📁 Project Structure

```
Computer-Vision-Application/
├── app.py                  # Flask application entry point
├── api_routes.py           # API endpoints for all modules
├── config.py               # Configuration management
├── error_handlers.py       # Error handling
├── validators.py           # Input validation
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container configuration
├── Procfile                # Process file for deployment
├── railway.toml            # Railway configuration
├── runtime.txt             # Python version specification
├── start.sh                # Startup script
├── modules/                # Computer Vision modules
│   ├── module1_dimension.py
│   ├── module2_template.py
│   ├── module3_features.py
│   ├── module4_sift_stitching.py
│   ├── module5_tracking.py
│   └── module7_pose_hand.py
├── templates/              # HTML templates
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── module1.html
│   ├── module2.html
│   ├── module3.html
│   ├── module4.html
│   ├── module5.html
│   └── module7.html
└── static/                 # CSS and assets
    └── css/
        └── module.css
```

---

## 🛠️ Technical Stack

**Backend:**
- Flask 3.0.3
- OpenCV 4.9 (headless)
- NumPy 1.26.4
- SciPy 1.12.0
- Scikit-image 0.23.2
- MediaPipe 0.10.21
- Gunicorn 21.2.0

**Frontend:**
- HTML5, CSS3, JavaScript
- Canvas API for image processing
- WebRTC for camera access

**Database:**
- SQLite3

**Deployment:**
- Docker
- Railway (auto-deployment)

---

## 📊 API Endpoints

### Module 1: Dimension Estimation
```
POST /api/module1/undistort
POST /api/module1/estimate
```

### Module 2: Template Matching
```
POST /api/module2/match
POST /api/module2/restore
```

### Module 3: Feature Detection
```
POST /api/module3/detect
POST /api/module3/segment
```

### Module 4: Image Stitching
```
POST /api/module4/stitch
```

### Module 5: Object Tracking
```
POST /api/module5/init
POST /api/module5/track
```

### Module 7: Pose & Hand Tracking
```
POST /api/module7/pose
POST /api/module7/hand
POST /api/module7/calibrate
```

---

## 🎯 Performance

- Real-time processing at 30+ FPS
- Optimized FLANN-based feature matching
- Efficient image stitching with RANSAC
- Low-latency API responses
- Progress tracking for long operations

---

## 🔧 Configuration

Environment variables (set in Railway):

```bash
FLASK_ENV=production
PORT=8080
SECRET_KEY=your-secret-key
```

---

## 📝 License

Educational project by Mahendra Krishna Koneru and Sai Leenath Jampala (2025)

---

## 🌐 Links

- **Live App:** [https://computervision.up.railway.app/](https://computervision.up.railway.app/)
- **GitHub:** [https://github.com/mahendrak070/Computer-Vision-Application](https://github.com/mahendrak070/Computer-Vision-Application)
- **Railway Dashboard:** [https://railway.app](https://railway.app)

---

**Deployed on Railway** | **Powered by Flask & OpenCV** | **© 2025**
