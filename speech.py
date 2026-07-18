import speech_recognition as sr
import threading
from queue import Queue, Empty
from bicep import (
    stop_event,
    display_lock,
    sense,
    show_mic,
    classify_intent,
    handle_intent,
    O, R,
)

# ─── Speech thread ─────────────────────────────────────────────────────────────
def speech_thread():
    recognizer = sr.Recognizer()
    audio_queue = Queue(maxsize=2)       # cap backlog — drop stale audio
    fail_count  = 0                      # consecutive failure counter

    def recognition_worker():
        """Runs in its own thread — does all STT + intent work, never blocks mic capture."""
        nonlocal fail_count
        while not stop_event.is_set():
            try:
                audio_data = audio_queue.get(timeout=1)
            except Empty:
                continue

            try:
                spoken = recognizer.recognize_google(audio_data)
                fail_count = 0
                print(f"[voice] Heard: {spoken}")
                intent_obj = classify_intent(spoken)
                print(f"[voice] Intent: {intent_obj}")
                handle_intent(intent_obj, spoken)

            except sr.UnknownValueError:
                fail_count += 1
                print(f"[voice] Could not understand audio (fail #{fail_count})")
                # Only show "?" after 2+ consecutive failures to reduce display noise
                if fail_count >= 2:
                    with display_lock:
                        sense.set_pixel(7, 7, O)   # single pixel — no scroll, no block
                show_mic()

            except sr.RequestError as e:
                fail_count += 1
                print(f"[voice] STT error: {e}")
                with display_lock:
                    sense.set_pixel(7, 7, R)
                show_mic()

            except Exception as e:
                print(f"[voice] Unexpected error: {e}")
                show_mic()

    # Start the worker as a daemon — it dies when main thread exits
    worker = threading.Thread(target=recognition_worker, daemon=True)
    worker.start()

    with sr.Microphone() as source:
        print("[voice] Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("[voice] Ready.")

        while not stop_event.is_set():
            try:
                show_mic()
                audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=8)

                # Drop audio if worker is already busy (queue full = stale capture)
                if not audio_queue.full():
                    audio_queue.put_nowait(audio_data)
                    with display_lock:
                        sense.set_pixel(0, 7, O)   # tiny "processing" dot, non-blocking

            except sr.WaitTimeoutError:
                show_mic()
            except Exception as e:
                print(f"[voice] Capture error: {e}")
                show_mic()
