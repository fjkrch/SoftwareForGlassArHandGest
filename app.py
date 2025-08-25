import cv2 as cv
import time
import threading
import queue
import platform
import subprocess
import sys

# --- IMPORTANT: import MediaPipe SOLUTIONS directly (won't pull TensorFlow) ---
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

# ----------------- constants -----------------
FINGER_TIPS = [4, 8, 12, 16, 20]
TIP_NAMES = {4:"Thumb", 8:"Index", 12:"Middle", 16:"Ring", 20:"Pinky"}
TOLERANCE = 0.10
CHECK_INTERVAL = 1.0  # speak check every 1 second

# -------------- TTS engines (no pyttsx3) --------------
class TTSEngineBase:
    def speak(self, text: str): raise NotImplementedError()
    def stop(self): pass

class WindowsSAPI(TTSEngineBase):
    def __init__(self, rate_delta=0, volume=100):
        import win32com.client  # pip install pywin32
        self.speaker = win32com.client.Dispatch("SAPI.SpVoice")
        try: self.speaker.Rate += int(rate_delta)
        except Exception: pass
        try: self.speaker.Volume = int(volume)
        except Exception: pass
    def speak(self, text: str):
        self.speaker.Speak(text)
    def stop(self):
        try: self.speaker = None
        except Exception: pass

class MacSay(TTSEngineBase):
    def __init__(self, voice=None, rate=180):
        self.voice = voice; self.rate = rate
    def speak(self, text: str):
        cmd = ["say"]
        if self.voice: cmd += ["-v", self.voice]
        if self.rate: cmd += ["-r", str(self.rate)]
        cmd.append(text)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class LinuxEspeak(TTSEngineBase):
    def __init__(self, voice=None, rate=180, volume=None):
        self.voice = voice; self.rate = rate; self.volume = volume
    def speak(self, text: str):
        cmd = ["espeak", f"-s{self.rate}"]
        if self.voice:  cmd += [f"-v{self.voice}"]
        if self.volume: cmd += [f"-a{int(self.volume)}"]  # 0..200
        cmd += [text]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class SpeechWorker:
    def __init__(self):
        self.q = queue.Queue()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.engine = None

    def start(self):
        system = platform.system().lower()
        try:
            if system == "windows":   self.engine = WindowsSAPI(rate_delta=0, volume=100)
            elif system == "darwin":  self.engine = MacSay(rate=170)
            else:                     self.engine = LinuxEspeak(rate=170)
        except Exception as e:
            print(f"[TTS] init error: {e}", file=sys.stderr)
            self.engine = None
        self.thread.start()

    def speak(self, text: str):
        if text and self.engine is not None:
            self.q.put(text)

    def stop(self):
        self._stop.set()
        self.q.put(None)
        self.thread.join(timeout=2.0)
        if self.engine:
            try: self.engine.stop()
            except Exception: pass
        self.engine = None

    def _run(self):
        while not self._stop.is_set():
            msg = self.q.get()
            if msg is None: break
            try:
                self.engine.speak(msg)
            except Exception as e:
                print(f"[TTS] speak error: {e}", file=sys.stderr)

# -------------- gesture helpers --------------
def relative_to_thumb(hand_lms):
    thumb = hand_lms.landmark[4]
    return [
        (hand_lms.landmark[idx].x - thumb.x,
         hand_lms.landmark[idx].y - thumb.y,
         hand_lms.landmark[idx].z - thumb.z)
        for idx in FINGER_TIPS
    ]

def within_tol(curr, ref, tol=TOLERANCE):
    if len(curr) != 5 or len(ref) != 5: return False
    for (cx,cy,cz),(rx,ry,rz) in zip(curr, ref):
        if abs(cx - rx) > tol: return False
        if abs(cy - ry) > tol: return False
        if abs(cz - rz) > tol: return False
    return True

# import gesture references from separate file
from gesture_refs import GESTURE_REFS

def classify_gesture(curr_tips):
    for name, ref in GESTURE_REFS.items():
        if within_tol(curr_tips, ref):
            return name
    return None

# -------------- main --------------
def main():
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # CAP_DSHOW helps on Windows
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    tts = SpeechWorker()
    tts.start()

    last_spoken = None
    last_check_time = 0.0
    prev_time = 0.0

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        ) as hands:

            while True:
                ok, frame = cap.read()
                if not ok:
                    continue  # keep looping instead of crashing

                frame = cv.flip(frame, 1)
                debug = frame.copy()

                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                results = hands.process(rgb)

                gesture = "No hand detected"
                tips_rel = None
                if results.multi_hand_landmarks:
                    hand_lms = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(debug, hand_lms, mp_hands.HAND_CONNECTIONS)

                    # draw tips (optional)
                    h, w = debug.shape[:2]
                    for idx in FINGER_TIPS:
                        lm = hand_lms.landmark[idx]
                        x_px, y_px = int(lm.x * w), int(lm.y * h)
                        cv.circle(debug, (x_px, y_px), 8, (0, 0, 255), -1)
                        cv.putText(debug, TIP_NAMES[idx], (x_px + 6, y_px - 6),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

                    tips_rel = relative_to_thumb(hand_lms)
                    g = classify_gesture(tips_rel)
                    if g: gesture = g

                # Speak check every second; only if new non-"no hand"
                now = time.time()
                if now - last_check_time >= CHECK_INTERVAL:
                    if gesture != "No hand detected" and gesture != last_spoken:
                        tts.speak(gesture)
                        last_spoken = gesture
                    last_check_time = now

                # UI
                cv.putText(debug, f"Gesture: {gesture}", (50, 100),
                           cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv.LINE_AA)

                # FPS
                curr = time.time()
                fps = 1 / (curr - prev_time) if prev_time else 0.0
                prev_time = curr
                cv.putText(debug, f'FPS: {int(fps)}', (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

                cv.imshow("Sign Speak (solutions import + SAPI TTS)", debug)
                key = cv.waitKey(1) & 0xFF

                # Press 's' to print relative-to-thumb coordinates in one-line tuple form
                if key == ord('s'):
                    if tips_rel:
                        print(",".join(f"({r[0]:.3f},{r[1]:.3f},{r[2]:.3f})" for r in tips_rel))
                    else:
                        print("No hand data")

                if key == 27:  # ESC
                    break

    except KeyboardInterrupt:
        pass
    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv.destroyAllWindows()
        tts.stop()

if __name__ == "__main__":
    main()
