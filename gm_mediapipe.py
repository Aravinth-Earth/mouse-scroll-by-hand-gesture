# pip install mediapipe opencv-python
# Copyright (C) 2026 Aravinth-Earth
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Gesture Mouse - MediaPipe Version
Uses Google's MediaPipe Hand Landmarker for accurate hand detection

References:
- https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb
"""
import ctypes
import math
import sys
import time
from ctypes import wintypes

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ULONG_PTR = getattr(wintypes, "ULONG_PTR", wintypes.WPARAM)

# ═══════════════════════════════════════════════════════════
# SCROLL CONFIGURATION - Adjust these values!
# ═══════════════════════════════════════════════════════════
CONTROL_MODE = "finger_direction"  # "finger_direction", "handedness", or "position"

# Finger direction mode: Point fingers up/down to scroll!
# Most intuitive - works with any hand

# Handedness mode: Just show left or right hand!
# Left hand = scroll down, Right hand = scroll up

# Position mode: Move hand up/down in frame

ACTIVATION_ZONE = 0.30  # Top/bottom 30% for position mode
SCROLL_SPEED = 2        # Scroll steps per frame (1=slow, 5=fast)

# Finger direction settings - ANGLE-BASED SPEED!
MAX_SCROLL_SPEED = 5         # Maximum scroll speed at 90 degrees
ANGLE_THRESHOLD = 15         # Minimum angle (degrees) to start scrolling (0-90)

# MediaPipe settings
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

WINDOW_TITLE = "Gesture Scroll - MediaPipe (ESC to quit)"


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class Input(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("mi", MouseInput)]


class WinScroller:
    INPUT_MOUSE = 0
    MOUSEEVENTF_WHEEL = 0x0800
    WHEEL_DELTA = 120

    def __init__(self):
        if sys.platform != "win32":
            raise SystemExit("This script is Windows-only.")
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.SendInput.argtypes = (
            wintypes.UINT,
            ctypes.POINTER(Input),
            ctypes.c_int,
        )
        self._user32.SendInput.restype = wintypes.UINT

    def scroll(self, steps):
        steps = int(steps)
        if steps == 0:
            return
        mouse_data = steps * self.WHEEL_DELTA
        inp = Input(
            type=self.INPUT_MOUSE,
            mi=MouseInput(0, 0, mouse_data, self.MOUSEEVENTF_WHEEL, 0, 0),
        )
        self._user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(Input))


def download_model():
    """Download the MediaPipe hand landmarker model if not present."""
    import os
    import urllib.request
    
    model_path = "hand_landmarker.task"
    
    if os.path.exists(model_path):
        return model_path
    
    print("📥 Downloading MediaPipe hand landmarker model (~10MB)...")
    print("   This runs 100% locally on your CPU - no cloud processing!")
    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print(f"✓ Model downloaded: {model_path}")
        print(f"  Size: ~10MB | CPU inference: 17ms/frame (~58 FPS)")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Download manually from:")
        print(model_url)
        sys.exit(1)


def get_finger_direction(hand_landmarks):
    """
    Detect finger pointing direction and calculate angle-based speed.
    Returns: (direction, speed_multiplier, angle_deg)
    - direction: 'up', 'down', or 'neutral'
    - speed_multiplier: 0.0 to 1.0 based on angle (0° horizontal = 0, 90° vertical = 1)
    - angle_deg: actual angle in degrees from horizontal
    
    Uses MediaPipe's 21 hand landmarks:
    - Index finger: tip=8, base=5
    - Middle finger: tip=12, base=9
    """
    # Get key landmarks (using normalized coordinates)
    index_tip = hand_landmarks[8]    # Index finger tip
    index_base = hand_landmarks[5]   # Index finger base (MCP)
    
    middle_tip = hand_landmarks[12]  # Middle finger tip
    middle_base = hand_landmarks[9]  # Middle finger base
    
    # Calculate vertical and horizontal differences
    index_dy = index_tip.y - index_base.y
    index_dx = index_tip.x - index_base.x
    middle_dy = middle_tip.y - middle_base.y
    middle_dx = middle_tip.x - middle_base.x
    
    # Average for more stable detection
    avg_dy = (index_dy + middle_dy) / 2
    avg_dx = (index_dx + middle_dx) / 2
    
    # Calculate angle from horizontal (0° = horizontal, 90° = vertical)
    # Use atan2 to get angle, taking absolute values for magnitude
    angle_rad = math.atan2(abs(avg_dy), abs(avg_dx))  # atan2(opposite, adjacent)
    angle_deg = math.degrees(angle_rad)
    
    # Calculate speed multiplier based on angle
    if angle_deg < ANGLE_THRESHOLD:
        # Too horizontal - no scroll
        speed_multiplier = 0.0
    else:
        # Linear mapping: ANGLE_THRESHOLD to 90° maps to 0.0 to 1.0
        speed_multiplier = (angle_deg - ANGLE_THRESHOLD) / (90 - ANGLE_THRESHOLD)
        speed_multiplier = min(1.0, max(0.0, speed_multiplier))  # Clamp to [0, 1]
    
    # Determine direction (negative y = pointing up in screen coordinates)
    if avg_dy < 0:  # Pointing up
        return 'up', speed_multiplier, angle_deg
    elif avg_dy > 0:  # Pointing down
        return 'down', speed_multiplier, angle_deg
    else:
        return 'neutral', 0.0, angle_deg


def draw_hand_landmarks(frame, detection_result):
    """
    Draw hand landmarks on the frame.
    Simple implementation without external drawing utilities.
    """
    annotated_image = np.copy(frame)
    h, w = frame.shape[:2]
    
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    
    if not hand_landmarks_list:
        return annotated_image
    
    # Define hand connections (21 landmarks connected)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # index
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
        (5, 9), (9, 13), (13, 17),  # palm
    ]
    
    # Draw each detected hand
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        
        # Get hand label and confidence
        hand_label = handedness[0].category_name
        hand_confidence = handedness[0].score
        
        # Color based on handedness
        landmark_color = (100, 255, 100) if hand_label == "Right" else (255, 100, 100)
        connection_color = (0, 200, 0) if hand_label == "Right" else (200, 0, 0)
        
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start, end = connection
            if start < len(hand_landmarks) and end < len(hand_landmarks):
                start_pos = hand_landmarks[start]
                end_pos = hand_landmarks[end]
                
                x1, y1 = int(start_pos.x * w), int(start_pos.y * h)
                x2, y2 = int(end_pos.x * w), int(end_pos.y * h)
                
                cv2.line(annotated_image, (x1, y1), (x2, y2), connection_color, 2)
        
        # Draw landmarks
        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(annotated_image, (x, y), 4, landmark_color, -1)
        
        # Draw hand label
        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]
        text_x = int(min(x_coords) * w)
        text_y = int(min(y_coords) * h) - 10
        
        cv2.putText(annotated_image, f"{hand_label} ({hand_confidence:.2f})",
                   (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, landmark_color, 2)
    
    return annotated_image


def main():
    if sys.platform != "win32":
        raise SystemExit("This script is Windows-only.")
    
    # Download model if needed
    model_path = download_model()
    
    # Initialize MediaPipe Hand Landmarker
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,  # Detect only one hand for simplicity
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
    
    scroller = WinScroller()

    # Open webcam
    if hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_print = time.time()

    print("═══════════════════════════════════════════════════════")
    print("  GESTURE SCROLL - MediaPipe Edition")
    print("═══════════════════════════════════════════════════════")
    print("")
    print("  ✅ 100% LOCAL PROCESSING - No cloud, no data sharing")
    print("  ⚡ CPU Usage: ~17ms per frame (58 FPS on CPU)")
    print("")
    if CONTROL_MODE == "finger_direction":
        print("  👆 FINGER DIRECTION MODE - Angle-based speed!")
        print("  ☝️  Point fingers UP    → Scroll UP")
        print("  👇 Point fingers DOWN  → Scroll DOWN")
        print("  ↔️  Horizontal fingers  → NO SCROLL")
        print("")
        print(f"  📐 Speed by angle: {ANGLE_THRESHOLD}° = start, 90° = max speed ({MAX_SCROLL_SPEED})")
        print("  Works with ANY hand - just point where you want to scroll!")
    elif CONTROL_MODE == "handedness":
        print("  👋 HANDEDNESS MODE - AI-powered left/right detection")
        print("  🤚 Show RIGHT hand → Scroll UP")
        print("  🖐️  Show LEFT hand  → Scroll DOWN")
    elif CONTROL_MODE == "position":
        print("  📍 POSITION MODE")
        print("  🖐️  Move hand to TOP    → Scroll UP")
        print("  🖐️  Move hand to BOTTOM → Scroll DOWN")
    print("")
    print(f"  ⚙️  Scroll Speed: {SCROLL_SPEED}")
    print("  📚 Based on Google's official MediaPipe HandLandmarker")
    print("  Press ESC to quit")
    print("═══════════════════════════════════════════════════════\n")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            frame_count = 0
            
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                now = time.time()
                should_print = (now - last_print) > 1.0
                frame_count += 1

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                # Convert to MediaPipe Image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                
                # Detect hands (timestamp in milliseconds)
                timestamp_ms = int(time.time() * 1000)
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                # Draw visual guides
                if CONTROL_MODE == "finger_direction":
                    # Draw finger direction indicator
                    center_y = h // 2
                    cv2.putText(frame, "☝️ POINT UP", (w//2 - 100, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
                    cv2.line(frame, (w//2, 50), (w//2, 80), (100, 255, 100), 3)
                    cv2.circle(frame, (w//2, 50), 8, (100, 255, 100), -1)
                    
                    cv2.putText(frame, "👇 POINT DOWN", (w//2 - 120, h - 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
                    cv2.line(frame, (w//2, h - 70), (w//2, h - 40), (255, 100, 100), 3)
                    cv2.circle(frame, (w//2, h - 40), 8, (255, 100, 100), -1)
                
                elif CONTROL_MODE == "handedness":
                    left_x = w // 4
                    right_x = 3 * w // 4
                    y_pos = 50
                    
                    cv2.putText(frame, "LEFT", (left_x - 40, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
                    cv2.putText(frame, "= DOWN", (left_x - 50, y_pos + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
                    
                    cv2.putText(frame, "RIGHT", (right_x - 50, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
                    cv2.putText(frame, "= UP", (right_x - 40, y_pos + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                
                elif CONTROL_MODE == "position":
                    zone_top = int(h * ACTIVATION_ZONE)
                    zone_bottom = int(h * (1 - ACTIVATION_ZONE))
                    cv2.line(frame, (0, zone_top), (w, zone_top), (0, 255, 255), 2)
                    cv2.line(frame, (0, zone_bottom), (w, zone_bottom), (0, 255, 255), 2)
                    cv2.putText(frame, "SCROLL UP", (10, zone_top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, "SCROLL DOWN", (10, zone_bottom + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Process detected hands
                if detection_result.hand_landmarks:
                    # Draw hand landmarks (official Google drawing)
                    frame = draw_hand_landmarks(frame, detection_result)
                    
                    # Get first hand
                    hand_landmarks = detection_result.hand_landmarks[0]
                    handedness = detection_result.handedness[0]
                    
                    # Get handedness (Left or Right)
                    hand_label = handedness[0].category_name
                    hand_confidence = handedness[0].score
                    
                    # Get wrist position (landmark 0)
                    wrist = hand_landmarks[0]
                    cx = int(wrist.x * w)
                    cy = int(wrist.y * h)
                    
                    # Determine scroll action
                    status = "IDLE"
                    scroll_steps = 0
                    
                    if CONTROL_MODE == "finger_direction":
                        # Finger direction mode - ANGLE-BASED SPEED!
                        finger_dir, speed_mult, angle = get_finger_direction(hand_landmarks)
                        
                        # Calculate actual scroll amount based on angle
                        scroll_amount = int(MAX_SCROLL_SPEED * speed_mult)
                        
                        if finger_dir == 'up' and scroll_amount > 0:
                            scroll_steps = scroll_amount
                            status = f"☝️ UP {int(angle)}°"
                            cv2.putText(frame, f"UP {int(angle)}deg - Speed {scroll_amount}", (w//2 - 160, h - 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 100), 3)
                            
                            # Draw arrow pointing up - size based on speed
                            arrow_len = int(40 + speed_mult * 80)
                            cv2.arrowedLine(frame, (cx, cy + arrow_len//2), (cx, cy - arrow_len//2), 
                                          (100, 255, 100), 5, tipLength=0.3)
                            
                        elif finger_dir == 'down' and scroll_amount > 0:
                            scroll_steps = -scroll_amount
                            status = f"👇 DOWN {int(angle)}°"
                            cv2.putText(frame, f"DOWN {int(angle)}deg - Speed {scroll_amount}", (w//2 - 180, h - 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 100), 3)
                            
                            # Draw arrow pointing down - size based on speed
                            arrow_len = int(40 + speed_mult * 80)
                            cv2.arrowedLine(frame, (cx, cy - arrow_len//2), (cx, cy + arrow_len//2), 
                                          (255, 100, 100), 5, tipLength=0.3)
                        else:
                            status = f"NEUTRAL {int(angle)}°"
                            cv2.putText(frame, f"Angle: {int(angle)}deg - Point UP/DOWN to scroll", (10, h - 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                            # Show small circle when neutral
                            cv2.circle(frame, (cx, cy), 15, (200, 200, 200), 3)
                        
                        if scroll_steps != 0:
                            scroller.scroll(scroll_steps)
                            if should_print:
                                print(f"[{status}] Angle: {angle:.1f}°, Speed mult: {speed_mult:.2f}, Scrolling {scroll_steps} steps")
                        elif should_print:
                            print(f"[{status}] Angle: {angle:.1f}° (< {ANGLE_THRESHOLD}° threshold)")
                    
                    elif CONTROL_MODE == "handedness":
                        # Use MediaPipe's accurate handedness detection!
                        if hand_label == "Right":
                            scroll_steps = SCROLL_SPEED
                            status = "🤚 RIGHT → UP"
                            cv2.putText(frame, "RIGHT HAND - SCROLLING UP", (w//2 - 180, h - 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 100), 3)
                            cv2.circle(frame, (cx, cy), 20, (100, 255, 100), 3)
                        elif hand_label == "Left":
                            scroll_steps = -SCROLL_SPEED
                            status = "🖐️ LEFT → DOWN"
                            cv2.putText(frame, "LEFT HAND - SCROLLING DOWN", (w//2 - 180, h - 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 100), 3)
                            cv2.circle(frame, (cx, cy), 20, (255, 100, 100), 3)
                        
                        if scroll_steps != 0:
                            scroller.scroll(scroll_steps)
                            if should_print:
                                print(f"[{status}] Detected: {hand_label} hand (confidence: {hand_confidence:.2f})")
                        elif should_print:
                            print(f"[READY] {hand_label} hand detected")
                    
                    elif CONTROL_MODE == "position":
                        # Position-based scrolling
                        norm_y = cy / h
                        
                        if norm_y < ACTIVATION_ZONE:
                            scroll_steps = SCROLL_SPEED
                            status = "↑ UP"
                            cv2.putText(frame, "SCROLLING UP", (w//2 - 100, 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                        elif norm_y > (1 - ACTIVATION_ZONE):
                            scroll_steps = -SCROLL_SPEED
                            status = "↓ DOWN"
                            cv2.putText(frame, "SCROLLING DOWN", (w//2 - 120, h - 20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                        else:
                            status = "READY"
                        
                        if scroll_steps != 0:
                            scroller.scroll(scroll_steps)
                            if should_print:
                                print(f"[{status}] Hand Y: {norm_y:.2f}, Scrolling {scroll_steps} steps")
                        elif should_print:
                            print(f"[{status}] Hand Y: {norm_y:.2f}")
                
                else:
                    if should_print:
                        print("[NO HAND] Show hand to camera")
                    cv2.putText(frame, "NO HAND DETECTED", (w//2 - 150, h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if should_print:
                    last_print = now

                cv2.imshow(WINDOW_TITLE, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
