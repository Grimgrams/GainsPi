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

# ─── Gemini setup ─────────────────────────────────────────────────────────────

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




# ─── Shared state ─────────────────────────────────────────────────────────────
@dataclass
class WorkoutState:
    mode: str = "bicep"

    # bicep counters
    bicep_right_count: int = 0
    bicep_right_up:    bool = False
    bicep_left_count:  int = 0
    bicep_left_up:     bool = False

    # press counters
    press_right_count: int = 0
    press_right_down:  bool = False
    press_left_count:  int = 0
    press_left_down:   bool = False

    lock: threading.Lock = field(default_factory=threading.Lock)

    def get_counts(self):
        with self.lock:
            if self.mode == "bicep":
                return self.bicep_left_count, self.bicep_right_count
            return self.press_left_count, self.press_right_count

    def reset_counts(self):
        with self.lock:
            if self.mode == "bicep":
                self.bicep_right_count = 0
                self.bicep_left_count  = 0
                self.bicep_right_up    = False
                self.bicep_left_up     = False
            else:
                self.press_right_count = 0
                self.press_left_count  = 0
                self.press_right_down  = False
                self.press_left_down   = False

    def set_mode(self, new_mode: str):
        with self.lock:
            self.mode = new_mode

    def get_mode(self):
        with self.lock:
            return self.mode


# ─── Globals ──────────────────────────────────────────────────────────────────
stop_event   = threading.Event()
display_lock = threading.Lock()
sense        = SenseHat()
state        = WorkoutState()

# ─── Display helpers ──────────────────────────────────────────────────────────
def scroll(text, colour=C):
    with display_lock:
        sense.show_message(text, scroll_speed=0.05, text_colour=colour)


def show_mode_icon(mode):
    with display_lock:
        sense.set_pixels(BICEP_ICON if mode == "bicep" else PRESS_ICON)
    time.sleep(1.5)


def show_wave_detected():
    with display_lock:
        sense.set_pixels(WAVE_ICON)
    time.sleep(0.6)
    with display_lock:
        sense.clear()


def show_mic():
    locked = display_lock.acquire(blocking=False)
    if locked:
        try:
            sense.set_pixels(MIC_ICON)
        finally:
            display_lock.release()


def _draw_counter(pixels, count_mod10, col_first, col_second, color):
    for i in range(5):
        row = 7 - i
        pixels[row * 8 + col_first]  = color if i < min(count_mod10, 5)     else OFF
        pixels[row * 8 + col_second] = color if i < max(count_mod10 - 5, 0) else OFF


def _draw_tally(pixels, count, col, color):
    marks = min(count // 10, 8)
    for i in range(8):
        row = 7 - i
        pixels[row * 8 + col] = color if i < marks else OFF


def _draw_rep_indicator(pixels, is_up, color):
    """
    Draws a 2-wide indicator in columns 3 & 4.
    Top 3 rows lit = arms UP, bottom 3 rows lit = arms DOWN.
    """
    if is_up:
        # Top of matrix = arms overhead
        active_rows = [0, 1, 2]
    else:
        # Bottom of matrix = arms at rest/down
        active_rows = [5, 6, 7]

    for row in range(8):
        on = row in active_rows
        pixels[row * 8 + 3] = color if on else OFF
        pixels[row * 8 + 4] = color if on else OFF


def display_bicep_count(left_count, right_count, left_up=False, right_up=False):
    locked = display_lock.acquire(blocking=False)
    if locked:
        try:
            pixels = [OFF] * 64
            _draw_counter(pixels, left_count % 10, 0, 1, G)
            _draw_tally(pixels, left_count, 2, G)
            _draw_counter(pixels, right_count % 10, 7, 6, C)
            _draw_tally(pixels, right_count, 5, C)
            # Show whichever arm is actively moving, prefer right
            is_up = right_up or left_up
            _draw_rep_indicator(pixels, is_up, W)
            sense.set_pixels(pixels)
        finally:
            display_lock.release()


def display_press_count(left_count, right_count, left_down=False, right_down=False):
    locked = display_lock.acquire(blocking=False)
    if locked:
        try:
            pixels = [OFF] * 64
            _draw_counter(pixels, left_count  % 10, col_first=0, col_second=1, color=Y)
            _draw_tally  (pixels, left_count,       col=2,                     color=Y)
            _draw_counter(pixels, right_count % 10, col_first=7, col_second=6, color=P)
            _draw_tally  (pixels, right_count,      col=5,                     color=P)
            # For press: down_flag = arms are in the DOWN (loaded) position
            is_up = not (right_down or left_down)
            _draw_rep_indicator(pixels, is_up, W)
            sense.set_pixels(pixels)
        finally:
            display_lock.release()


# ─── Maths ────────────────────────────────────────────────────────────────────
def calculate_angle(a, b, c):
    a  = np.array(a);  b = np.array(b);  c = np.array(c)
    ba = a - b;        bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


# ─── Interpreter ──────────────────────────────────────────────────────────────
def make_interpreter(model_path):
    try:
        interp = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")]
        )
        print("[INFO] Running on Edge TPU")
        return interp
    except (ValueError, OSError) as e:
        print(f"[WARN] Edge TPU unavailable ({e}), falling back to CPU")
        cpu_model = model_path.replace("_edgetpu.tflite", ".tflite")
        return tflite.Interpreter(model_path=cpu_model)


def input_size(interpreter):
    _, h, w, _ = interpreter.get_input_details()[0]["shape"]
    return w, h


def get_output(interpreter):
    out = interpreter.get_output_details()[0]
    return interpreter.get_tensor(out["index"])


# ─── Gesture detector ─────────────────────────────────────────────────────────
class GestureDetector:
    def __init__(self):
        self._bicep_frames = 0
        self._press_frames = 0
        self._last_trigger = 0.0

    def _arm_extended_and_down(self, kp, sh, el, wr, hip):
        sx, sy, sc = kp[sh];  ex, ey, ec = kp[el]
        wx, wy, wc = kp[wr];  hx, hy, hc = kp[hip]
        if sc < 0.3 or ec < 0.3 or wc < 0.3 or hc < 0.3:
            return False
        return calculate_angle((sx, sy), (ex, ey), (wx, wy)) > ARM_EXTENDED_ANGLE and wy > hy

    def _press_pose(self, kp, sh, el, wr):
        sx, sy, sc = kp[sh];  ex, ey, ec = kp[el];  wx, wy, wc = kp[wr]
        if sc < 0.3 or ec < 0.3 or wc < 0.3:
            return False
        return wy < sy and ey < sy

    def update(self, keypoints, current_mode):
        now = time.time()
        if now - self._last_trigger < GESTURE_COOLDOWN:
            return current_mode

        seeing_bicep = (
            self._arm_extended_and_down(keypoints, LEFT_SHOULDER,  LEFT_ELBOW,  LEFT_WRIST,  LEFT_HIP) and
            self._arm_extended_and_down(keypoints, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HIP)
        )
        seeing_press = (
            self._press_pose(keypoints, LEFT_SHOULDER,  LEFT_ELBOW,  LEFT_WRIST) or
            self._press_pose(keypoints, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        )

        if seeing_bicep and current_mode != "bicep":
            self._bicep_frames += 1
        else:
            self._bicep_frames = 0

        if seeing_press and current_mode != "press":
            self._press_frames += 1
        else:
            self._press_frames = 0

        if self._bicep_frames >= GESTURE_FRAMES_NEEDED:
            self._bicep_frames = self._press_frames = 0
            self._last_trigger = now
            return "bicep"
        if self._press_frames >= GESTURE_FRAMES_NEEDED:
            self._bicep_frames = self._press_frames = 0
            self._last_trigger = now
            return "press"
        return current_mode


# ─── Exercise counters ────────────────────────────────────────────────────────
def count_bicep(keypoints, ws: WorkoutState):
    with ws.lock:
        for side, sh, el, wr, up_key, count_key in [
            ("right", RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, "bicep_right_up", "bicep_right_count"),
            ("left",  LEFT_SHOULDER,  LEFT_ELBOW,  LEFT_WRIST,  "bicep_left_up",  "bicep_left_count"),
        ]:
            sx, sy, sc = keypoints[sh]
            ex, ey, ec = keypoints[el]
            wx, wy, wc = keypoints[wr]
            if sc > 0.3 and ec > 0.3 and wc > 0.3:
                angle = calculate_angle((sx, sy), (ex, ey), (wx, wy))
                if angle < CURL_UP_ANGLE:
                    setattr(ws, up_key, True)
                if angle > CURL_DOWN_ANGLE and getattr(ws, up_key):
                    setattr(ws, count_key, getattr(ws, count_key) + 1)
                    setattr(ws, up_key, False)
        return ws.bicep_left_count, ws.bicep_right_count


def count_shoulder_press(keypoints, ws: WorkoutState):
    with ws.lock:
        for side, sh, el, wr, down_key, count_key in [
            ("right", RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, "press_right_down", "press_right_count"),
            ("left",  LEFT_SHOULDER,  LEFT_ELBOW,  LEFT_WRIST,  "press_left_down",  "press_left_count"),
        ]:
            sx, sy, sc = keypoints[sh]
            ex, ey, ec = keypoints[el]
            wx, wy, wc = keypoints[wr]

            if sc > 0.3 and ec > 0.3 and wc > 0.3:

                # dynamic threshold based on arm length
                arm_len = abs(sy - ey)

                down_threshold = sy + 0.05 * arm_len   # was 0.2–0.3
                up_threshold   = sy - 0.15 * arm_len   # was 0.3–0.5

                BUFFER = 10  # pixels

                if wy > down_threshold - BUFFER:
                    setattr(ws, down_key, True)

                angle = calculate_angle((sx, sy), (ex, ey), (wx, wy))

                if wy < up_threshold + BUFFER and angle > 110 and getattr(ws, down_key):
                    setattr(ws, count_key, getattr(ws, count_key) + 1)
                    setattr(ws, down_key, False)

        return ws.press_left_count, ws.press_right_count

# ─── Overlay ──────────────────────────────────────────────────────────────────
def draw_overlay(frame, mode, r_count, l_count, fps, keypoints):
    h, w, _ = frame.shape
    pairs = [
        (LEFT_SHOULDER, LEFT_ELBOW),   (LEFT_ELBOW,  LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
        (LEFT_SHOULDER, RIGHT_SHOULDER),
    ]
    for a, b in pairs:
        ax, ay, ac = keypoints[a];  bx, by, bc = keypoints[b]
        if ac > 0.3 and bc > 0.3:
            cv2.line(frame, (ax, ay), (bx, by), (200, 200, 200), 2)

    label = "BICEP CURL" if mode == "bicep" else "SHOULDER PRESS"
    cv2.putText(frame, label,                (20, 40),    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    cv2.putText(frame, f"FPS: {int(fps)}",   (20, 75),    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),   2)
    cv2.putText(frame, f"Right: {r_count}",  (20, 115),   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
    cv2.putText(frame, f"Left:  {l_count}",  (20, 150),   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 3)
    cv2.putText(frame, "Wave or speak to switch exercise",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


# ─── Gemini intent pipeline ───────────────────────────────────────────────────

# System prompt for the intent classifier.
# The model must return ONLY a JSON object — no markdown, no extra text.
INTENT_SYSTEM = """
You are an intent classifier for a voice-controlled fitness tracker running on a
Raspberry Pi with a SenseHat. The user speaks commands while working out.

Classify the utterance into EXACTLY ONE of these intents and return a JSON object:

  {"intent": "SENSOR_TEMP"}
      User is asking about temperature, heat, cold, warmth, or the room climate.
      Examples: "Is it hot in here?", "Tell me the temperature", "What temp is it?"

  {"intent": "SENSOR_HUMIDITY"}
      User is asking about humidity, moisture, sweat-related air quality.
      Examples: "How humid is it?", "Is the air damp?", "What is the humidity?"

  {"intent": "SENSOR_PRESSURE"}
      User is asking about air pressure, barometric pressure, elevation.
      Examples: "What's the pressure?", "How is the barometric pressure?"

  {"intent": "SENSOR_ALL"}
      User wants a general environment / sensor overview.
      Examples: "What are the sensor readings?", "How is the environment?",
                "Give me all the stats", "What does the sense hat say?"

  {"intent": "MODE_SWITCH", "mode": "bicep"}
      User wants to switch to bicep curl mode.
      Examples: "Switch to bicep curls", "Let's do curls", "Bicep mode",
                "Change to curls", "I want to do bicep curls now"

  {"intent": "MODE_SWITCH", "mode": "press"}
      User wants to switch to shoulder press mode.
      Examples: "Switch to shoulder press", "Let's do presses", "Press mode",
                "Change exercise to shoulder press", "Overhead press now"

  {"intent": "REP_QUERY"}
      User is asking how many reps they have done.
      Examples: "How many reps?", "What's my count?", "How am I doing?",
                "How many have I done?", "Tell me my rep count"

  {"intent": "RESET_REPS"}
      User wants to reset / clear the rep counters.
      Examples: "Reset my reps", "Start over", "Zero the counter",
                "Clear my count", "Reset everything"

  {"intent": "FREEFORM"}
      Anything that does not fit the above.

Return ONLY the JSON object. No markdown. No explanation.
""".strip()


def classify_intent(spoken: str) -> dict:
    """Ask Gemini to classify the spoken text and return a parsed dict."""
    prompt = f'{INTENT_SYSTEM}\n\nUtterance: "{spoken}"'
    try:
        resp = gemini.generate_content(prompt)
        raw  = resp.text.strip()
        # Strip accidental markdown fences
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$",        "", raw)
        return json.loads(raw)
    except Exception as e:
        print(f"[intent] Classification error: {e}")
        return {"intent": "FREEFORM"}


def get_sensor_context() -> str:
    temp     = round(sense.get_temperature(), 1)
    humidity = round(sense.get_humidity(),    1)
    pressure = round(sense.get_pressure(),    1)
    return (
        f"SenseHat readings — Temperature: {temp}°C, "
        f"Humidity: {humidity}%, Pressure: {pressure} mbar. "
    )


def trim(text: str, max_words: int = 30) -> str:
    words = text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


def handle_intent(intent_obj: dict, spoken: str):
    """
    Route the classified intent to the correct handler.
    Each handler may read sensors, update WorkoutState, and scroll the LED.
    """
    intent = intent_obj.get("intent", "FREEFORM")

    # ── Sensor intents ────────────────────────────────────────────────────────
    if intent == "SENSOR_TEMP":
        temp = round(sense.get_temperature(), 1)
        prompt = (
            f"The current room temperature is {temp}°C. "
            f"The user asked: \"{spoken}\". "
            f"Answer in one short sentence."
        )
        answer = trim(gemini.generate_content(prompt).text)
        print(f"[voice] {answer}")
        scroll(answer, colour=C)

    elif intent == "SENSOR_HUMIDITY":
        hum = round(sense.get_humidity(), 1)
        prompt = (
            f"The current humidity is {hum}%. "
            f"The user asked: \"{spoken}\". "
            f"Answer in one short sentence."
        )
        answer = trim(gemini.generate_content(prompt).text)
        print(f"[voice] {answer}")
        scroll(answer, colour=C)

    elif intent == "SENSOR_PRESSURE":
        pres = round(sense.get_pressure(), 1)
        prompt = (
            f"The current air pressure is {pres} mbar. "
            f"The user asked: \"{spoken}\". "
            f"Answer in one short sentence."
        )
        answer = trim(gemini.generate_content(prompt).text)
        print(f"[voice] {answer}")
        scroll(answer, colour=C)

    elif intent == "SENSOR_ALL":
        ctx    = get_sensor_context()
        prompt = (
            f"{ctx}"
            f"The user asked: \"{spoken}\". "
            f"Give a brief two-sentence summary of the environment."
        )
        answer = trim(gemini.generate_content(prompt).text, max_words=40)
        print(f"[voice] {answer}")
        scroll(answer, colour=C)

    # ── Mode switch ───────────────────────────────────────────────────────────
    elif intent == "MODE_SWITCH":
        new_mode = intent_obj.get("mode", "bicep")
        current  = state.get_mode()
        if new_mode == current:
            scroll(f"Already in {new_mode} mode", colour=O)
        else:
            state.set_mode(new_mode)
            # Reset flags for the new mode
            state.reset_counts()
            show_mode_icon(new_mode)
            scroll(f"Switched to {new_mode}", colour=G)
        print(f"[voice] Mode → {new_mode}")

    # ── Rep query ─────────────────────────────────────────────────────────────
    elif intent == "REP_QUERY":
        left, right = state.get_counts()
        mode        = state.get_mode()
        msg = f"{mode}: L{left} R{right}"
        print(f"[voice] {msg}")
        scroll(msg, colour=Y)

    # ── Reset reps ────────────────────────────────────────────────────────────
    elif intent == "RESET_REPS":
        state.reset_counts()
        scroll("Reps reset", colour=O)
        print("[voice] Reps reset")

    # ── Freeform fallback ─────────────────────────────────────────────────────
    else:
        prompt  = f"Answer in one or two short sentences only: {spoken}"
        answer  = trim(gemini.generate_content(prompt).text)
        print(f"[voice] {answer}")
        scroll(answer, colour=C)




# ─── Pose thread (main loop) ──────────────────────────────────────────────────
def pose_loop():
    model_path  = "movenet_single_pose_lightning_ptq_edgetpu.tflite"
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    input_width, input_height = input_size(interpreter)
    input_index = interpreter.get_input_details()[0]["index"]

    cap     = cv2.VideoCapture(0)
    pTime   = 0.0
    gesture = GestureDetector()

    show_mode_icon(state.get_mode())

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Pre-process
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = np.asarray(
            Image.fromarray(rgb).resize((input_width, input_height), Image.LANCZOS),
            dtype=np.uint8
        )
        interpreter.set_tensor(input_index, [resized])
        interpreter.invoke()

        pose      = get_output(interpreter).reshape(_NUM_KEYPOINTS, 3)
        keypoints = [
            (int(pose[i][1] * w), int(pose[i][0] * h), float(pose[i][2]))
            for i in range(_NUM_KEYPOINTS)
        ]

        for (x, y, conf) in keypoints:
            if conf > 0.3:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # ── Gesture mode switch ────────────────────────────────────────────
        current = state.get_mode()
        new_mode = gesture.update(keypoints, current)
        if new_mode != current:
            state.set_mode(new_mode)
            show_mode_icon(new_mode)

        # ── Count reps ────────────────────────────────────────────────────
        mode = state.get_mode()
        if mode == "bicep":
            l, r = count_bicep(keypoints, state)
            display_bicep_count(l, r,
                                left_up=state.bicep_left_up,
                                right_up=state.bicep_right_up)
        else:
            l, r = count_shoulder_press(keypoints, state)
            display_press_count(l, r,
                                left_down=state.press_left_down,
                                right_down=state.press_right_down)
        # ── FPS + overlay ─────────────────────────────────────────────────
        cTime = time.time()
        fps   = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        draw_overlay(frame, mode, r, l, fps, keypoints)

        cv2.imshow("Fitness Assistant", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()
