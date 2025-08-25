import cv2 as cv
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Thumb, Index, Middle, Ring, Pinky landmark indices
FINGER_TIPS = [4, 8, 12, 16, 20]
TIP_NAMES = {4:"Thumb", 8:"Index", 12:"Middle", 16:"Ring", 20:"Pinky"}

TOLERANCE = 0.07# ±20%

def relative_to_thumb(hand_lms):
    """Return fingertip coords relative to thumb tip (Lm 4)."""
    thumb = hand_lms.landmark[4]
    tips_rel = []
    for idx in FINGER_TIPS:
        lm = hand_lms.landmark[idx]
        tips_rel.append((lm.x - thumb.x, lm.y - thumb.y, lm.z - thumb.z))
    return tips_rel

def within_tol(curr, ref, tol=TOLERANCE):
    if len(curr) != 5 or len(ref) != 5:
        return False
    for (cx,cy,cz), (rx,ry,rz) in zip(curr, ref):
        if abs(cx - rx) > tol: return False
        if abs(cy - ry) > tol: return False
        if abs(cz - rz) > tol: return False
    return True

# Example reference gesture (STOP) using thumb as origin
# You should press 'S' to capture your own references
GESTURE_REFS = {
    "STOP": [
          (0.000,0.000,0.000),
    (0.018,-0.139,-0.000),
    (0.034,-0.169,-0.002),
    (0.069,-0.147,-0.006),
    (0.097,-0.084,-0.007),
    ],
    "OKAY": [(0.000,0.000,0.000),
    (0.009,-0.041,-0.027),
    (0.024,-0.243,-0.011),
    (0.078,-0.275,-0.005),
    (0.159,-0.249,-0.006),],
    "PAIN": [
   (0.000,0.000,0.000),
    (-0.018,0.050,-0.003),
    (0.006,0.037,0.017),
    (0.034,0.040,0.026),
    (0.060,0.013,0.035),],
    "Hurt a lot": [
    (0.000,0.000,0.000),
    (0.035,-0.308,-0.002),
    (0.117,-0.038,0.007),
    (0.141,-0.030,0.019),
    (0.159,-0.025,0.004),],
    "Hurt a a little": [
    (0.000,0.000,0.000),
    (0.005,-0.069,-0.002),
    (0.112,0.032,0.010),
    (0.127,0.042,0.021),
    (0.136,0.051,0.002),],
}

def classify_gesture(curr_tips):
    for name, ref in GESTURE_REFS.items():
        if within_tol(curr_tips, ref, tol=TOLERANCE):
            return name
    return None

def print_tips_relative(hand_lms):
    tips = relative_to_thumb(hand_lms)
    print("Captured tips relative to Thumb (thumb=origin):")
    for (x,y,z) in tips:
        print(f"    ({x:.3f},{y:.3f},{z:.3f}),")

def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)
    prev_time = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv.flip(frame, 1)
            debug = frame.copy()

            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture_label = None

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(debug, hand_lms, mp_hands.HAND_CONNECTIONS)

                    # relative fingertip coords
                    tips_rel = relative_to_thumb(hand_lms)

                    # classify
                    gesture_label = classify_gesture(tips_rel)

                    # Draw fingertips
                    h,w = debug.shape[:2]
                    for idx in FINGER_TIPS:
                        lm = hand_lms.landmark[idx]
                        x_px,y_px = int(lm.x*w), int(lm.y*h)
                        cv.circle(debug,(x_px,y_px),8,(0,0,255),-1)
                        cv.putText(debug,TIP_NAMES[idx],(x_px+6,y_px-6),
                                   cv.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv.LINE_AA)

            if gesture_label:
                cv.putText(debug,f"Gesture: {gesture_label}",(50,100),
                           cv.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3,cv.LINE_AA)

            # FPS
            curr = time.time()
            fps = 1 / (curr-prev_time) if prev_time else 0.0
            prev_time = curr
            cv.putText(debug,f'FPS: {int(fps)}',(10,30),
                       cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv.LINE_AA)

            cv.imshow("Relative to Thumb", debug)
            key = cv.waitKey(1) & 0xFF
            if key == 27: break
            elif key in (ord('s'),ord('S')):
                if results.multi_hand_landmarks:
                    print_tips_relative(results.multi_hand_landmarks[0])
                else:
                    print("No hands detected.")

    cap.release()
    cv.destroyAllWindows()

if __name__=="__main__":
    main()
