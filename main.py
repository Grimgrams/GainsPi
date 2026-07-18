"""
Fitness Virtual Assistant — bicep.py
CENG260 — Julian Pitterson  N01680049

Voice pipeline
──────────────
1. Google STT  →  raw spoken text
2. Gemini #1   →  intent JSON  (classify what the user wants)
3. Route intent:
     SENSOR_TEMP / SENSOR_HUMIDITY / SENSOR_PRESSURE / SENSOR_ALL
       → read SenseHat sensors, build a prompt, ask Gemini #2 for
         a natural 1-sentence answer, scroll it on the LED matrix.
     MODE_SWITCH (bicep | press)
       → update shared WorkoutState.mode, flash mode icon.
     REP_QUERY
       → report current rep counts via LED scroll.
     RESET_REPS
       → zero the rep counters in shared state.
     FREEFORM
       → send straight to Gemini #2 for a short general answer.

Shared state between the pose thread and the speech thread is held
in a WorkoutState dataclass protected by a threading.Lock.
"""

import json
import re
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import speech_recognition as sr
import google.generativeai as genai
from PIL import Image
from sense_hat import SenseHat
import tflite_runtime.interpreter as tflite
from bicep import *
from speech import *

# ─── Gemini setup ────────────────────────────────────────────────────────────
genai.configure(api_key="")
gemini = genai.GenerativeModel("gemini-2.5-flash")

# ─── Keypoint indices ─────────────────────────────────────────────────────────
_NUM_KEYPOINTS  = 17
NOSE            = 0
LEFT_SHOULDER   = 5
RIGHT_SHOULDER  = 6
LEFT_ELBOW      = 7
RIGHT_ELBOW     = 8
LEFT_WRIST      = 9
RIGHT_WRIST     = 10
LEFT_HIP        = 11
RIGHT_HIP       = 12

# ─── Thresholds ───────────────────────────────────────────────────────────────
CURL_UP_ANGLE         = 65
CURL_DOWN_ANGLE       = 130
PRESS_UP_ANGLE        = 160
PRESS_DOWN_ANGLE      = 90
GESTURE_FRAMES_NEEDED = 10
GESTURE_COOLDOWN      = 3.0
ARM_EXTENDED_ANGLE    = 155

# ─── Colours ──────────────────────────────────────────────────────────────────
G   = (0,   255,   0)
W   = (255, 255, 255)
_   = (0,     0,   0)
R   = (200,   0,   0)
C   = (0,   200, 255)
O   = (255, 140,   0)
Y   = (255, 255,   0)
P   = (180,   0, 255)
OFF = (0,     0,   0)

# ─── SenseHat icons ───────────────────────────────────────────────────────────
BICEP_ICON = [
    _, _, _, _, _, _, _, _,
    _, W, _, _, _, _, W, _,
    _, W, W, _, _, W, W, _,
    _, _, W, W, W, W, _, _,
    _, _, W, W, W, W, _, _,
    _, _, W, W, W, W, _, _,
    _, _, W, _, _, W, _, _,
    _, _, _, _, _, _, _, _,
]

PRESS_ICON = [
    _, _, _, W, W, _, _, _,
    _, _, W, W, W, W, _, _,
    _, W, _, W, W, _, W, _,
    _, _, _, W, W, _, _, _,
    _, _, _, W, W, _, _, _,
    _, _, _, W, W, _, _, _,
    _, _, W, W, W, W, _, _,
    _, _, _, _, _, _, _, _,
]

WAVE_ICON = [
    _, W, _, W, _, W, _, _,
    W, _, W, _, W, _, W, _,
    _, W, _, W, _, W, _, W,
    _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _,
]

MIC_ICON = [
    _, _, _, W, W, _, _, _,
    _, _, W, W, W, W, _, _,
    _, _, W, W, W, W, _, _,
    _, _, W, W, W, W, _, _,
    _, _, _, W, W, _, _, _,
    _, _, W, W, W, W, _, _,
    _, _, _, _, _, _, _, _,
    _, _, _, W, W, _, _, _,
]



# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    print("[main] Starting voice thread...")
    voice = threading.Thread(target=speech_thread, daemon=True)
    voice.start()

    print("[main] Starting pose loop...")
    try:
        pose_loop()
    except KeyboardInterrupt:
        print("[main] Interrupted")
    finally:
        stop_event.set()
        sense.clear()
        print("[main] Done.")


if __name__ == "__main__":
    main()