# AI-Based Smart Attendance System Using Face Recognition

A college mini-project prototype using Python, OpenCV, Tkinter and CSV attendance storage.

## Features
- Student registration
- Webcam face detection
- LBPH face recognition
- Automatic attendance with date and time
- Duplicate attendance prevention for the same day
- Attendance report

## Requirements
- Python 3.9+
- Webcam
- Windows/Linux/macOS

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Demo Steps
1. Click **Register Student** and enter register number and name.
2. Look at the camera until about 20 face samples are collected.
3. Click **Train Face Recognition Model**.
4. Click **Start Attendance**.
5. Stand in front of the camera. A registered student will be marked Present.
6. Click **View Attendance Report** to show the saved records.

## Notes
- This is a prototype for academic demonstration.
- Use sample/demo student data where possible.
- Obtain appropriate consent before collecting real facial data.
- Do not upload face images, biometric data, or private student information to a public repository.
