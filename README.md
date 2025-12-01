# Computer Vision Project Website

**Authors:** Mahendra Krishna Koneru and Sai Leenath Jampala

A comprehensive web-based computer vision platform with face authentication, real-time tracking, image processing, and advanced CV algorithms.

---

## Quick Start

### Installation

1. **Navigate to project:**
```bash
cd "/Users/mahi/Desktop/Computer Vision/WEB"
```

2. **Run startup script:**
```bash
# macOS/Linux
./start.sh

# Windows
start.bat
```

3. **Access application:**
```
http://localhost:5001
```

---

## Features

- **Face Authentication** - Secure face-based login and registration
- **Module 1** - Real-world dimension estimation
- **Module 2** - Template matching & Fourier restoration
- **Module 3** - Feature detection & segmentation
- **Module 4** - Image stitching & SIFT
- **Module 5-6** - Real-time object tracking
- **Module 7** - Pose estimation & hand tracking

---

## Usage

### First Time Setup

1. **Register:** http://localhost:5001/register
   - Enter username and email
   - Capture your face
   - Click Register

2. **Login:** http://localhost:5001/login
   - Click "Authenticate with Face"
   - Auto-capture and login

3. **Quick Access:** http://localhost:5001/dev-login
   - Bypass authentication for testing

### Access Modules

**Dashboard:** http://localhost:5001/dashboard

All 7 modules accessible from dashboard with intuitive interfaces.

---

## Troubleshooting

**Port already in use:**
```bash
lsof -i:5001
kill -9 <PID>
```

**Camera not working:**
- Allow browser camera permissions
- Close other apps using camera

**Face authentication fails:**
- Ensure good lighting
- Position face clearly
- Use dev-login for testing

---

## Technical Stack

- **Backend:** Flask, OpenCV, Mediapipe, face_recognition
- **Frontend:** HTML5, CSS3, JavaScript
- **Database:** SQLite
- **Server:** Flask (port 5001)

---

## Project Structure

```
WEB/
├── app.py              # Main Flask app
├── api_routes.py       # API endpoints
├── config.py           # Configuration
├── error_handlers.py   # Error handling
├── validators.py       # Input validation
├── modules/            # CV algorithms
├── templates/          # HTML templates
├── static/            # CSS/JS files
├── database/          # SQLite DB
└── face_encodings/    # Face data
```

---

## Server Commands

```bash
# Start server
./start.sh

# Stop server
pkill -9 -f "python app.py"

# Check status
lsof -i:5001

# View logs
tail -f server.log
```

---

## Health Check

```bash
curl http://localhost:5001/health
curl http://localhost:5001/api/status
```

---

## Contact

**Authors:** Mahendra Krishna Koneru and Sai Leenath Jampala  
**Version:** 1.0  
**Year:** 2025
