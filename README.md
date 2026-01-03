# Gesture Mouse Scroll (MediaPipe)


Local, CPU-only tool to control mouse scrolling with hand gestures via your webcam. This repository contains `gm_mediapipe.py`, which uses Google's MediaPipe Hand Landmarker and Windows SendInput for scrolling.

Author: Aravinth-Earth

**Platform**: Windows-only (script exits on non-win32 platforms)

**Key features (implemented in `gm_mediapipe.py`)**
- 3 control modes: `finger_direction` (default), `handedness`, and `position`.
- Angle-based speed in `finger_direction` mode (maps finger angle to scroll speed).
- Automatic download of `hand_landmarker.task` model on first run (script downloads from the official MediaPipe model URL).
- Uses native Windows `SendInput` to send mouse-wheel events (no external automation tools).
- Visual debugging overlay: hand landmarks, arrows, status text and activation zones.
- Prints concise runtime status to console (including approximate CPU timing printed by the script: ~17ms/frame in the printed message).

Quick reference of defaults (match the code)
- `CONTROL_MODE` = `finger_direction`
- `ACTIVATION_ZONE` = `0.30` (used by `position` mode)
- `SCROLL_SPEED` = `2`
- `MAX_SCROLL_SPEED` = `5` (used to scale angle->speed)
- `ANGLE_THRESHOLD` = `15` (degrees; minimum angle to start scroll)
- `MIN_DETECTION_CONFIDENCE` = `0.5`
- `MIN_TRACKING_CONFIDENCE` = `0.5`
- Window title shown by the script: `Gesture Scroll - MediaPipe (ESC to quit)`

Requirements
- Python 3.10+ (virtualenv recommended)
- Webcam
- `requirements.txt` (install with pip)

Quick setup

1. (Optional) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

Run

```powershell
python gm_mediapipe.py
```

What the script does on first run
- If `hand_landmarker.task` is not present, the script downloads it from:
	`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
- The script prints progress messages like "Downloading MediaPipe hand landmarker model..." and exits with a message and URL if download fails.

Modes and behavior
- `finger_direction` (default): detects index+middle finger direction and maps the vertical angle to scroll speed. Point up → scroll up; point down → scroll down. Angle below `ANGLE_THRESHOLD` is considered neutral (no scroll).
- `handedness`: uses MediaPipe handedness result. Right hand → scroll up; Left hand → scroll down.
- `position`: scrolls when the wrist Y position is above/below the activation zones (`ACTIVATION_ZONE`).

Runtime messages and UI hints (from code)
- Console prints: status lines like `[UP] Angle: 60.0°, Scrolling 3 steps` or `[NO HAND] Show hand to camera` approximately once per second.
- On-screen overlays: landmarks, arrows, labels, activation zone lines, and helpful on-screen text for each mode.

Troubleshooting
- "No camera detected" / black frames: ensure webcam is connected and not used by another process.
- "Model download failed": internet is required only for the initial model download; retry or download the model manually from the URL above and place `hand_landmarker.task` next to the script.
- If scrolling doesn't work: verify Windows permissions and that no security software blocks synthetic input. Ensure you're running on Windows and not WSL.
- "Hand not detected": improve lighting, move the hand into the camera view, or reduce camera resolution for performance tweaks in `gm_mediapipe.py`.

Development notes
- Main script: [gm_mediapipe.py](gm_mediapipe.py)
- Model file (downloaded at runtime): `hand_landmarker.task`

Possible improvements I can add on request
- Command-line flags for `CONTROL_MODE`, camera index, `SCROLL_SPEED`, `ANGLE_THRESHOLD`, and debug toggles.
- Option to log to a file, or a UI toggle to enable/disable scrolling without quitting.

If you want, I can update `gm_mediapipe.py` to accept CLI flags and add a short example usage section.