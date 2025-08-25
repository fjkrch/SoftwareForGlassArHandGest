import cv2 as cv
import numpy as np
import mediapipe as mp
import os
import time
import argparse
import copy

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# -----------------------------
# Reference pose (normalized x,y,z) from your example
# You can press 'R' at runtime to overwrite this with a fresh capture.
# -----------------------------
DEFAULT_REF = [
    (0.897,0.881,0.000), (0.828,0.855,-0.021), (0.772,0.781,-0.027),
    (0.746,0.702,-0.031), (0.733,0.634,-0.034), (0.807,0.618,-0.004),
    (0.785,0.511,-0.015), (0.775,0.446,-0.027), (0.769,0.389,-0.037),
    (0.846,0.597,-0.006), (0.829,0.476,-0.014), (0.822,0.400,-0.025),
    (0.817,0.337,-0.034), (0.886,0.602,-0.013), (0.880,0.488,-0.026),
    (0.878,0.416,-0.036), (0.876,0.353,-0.044), (0.928,0.627,-0.023),
    (0.937,0.543,-0.037), (0.943,0.489,-0.042), (0.947,0.437,-0.046),
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument("--input", help='path to image or video file (optional)', type=str, default='')
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=0.10,
                        help="Per-axis tolerance for x,y,z (normalized). Example 0.10 = 10%%")
    return parser.parse_args()

def landmarks_to_list(lms):
    """Convert MediaPipe landmarks to list of (x,y,z) tuples (normalized)."""
    return [(lm.x, lm.y, lm.z) for lm in lms]

def all_within_tolerance(curr, ref, tol=0.10):
    """
    Return True if for every landmark index i, abs(curr[i].axis - ref[i].axis) <= tol
    for axis in {x,y,z}. Uses normalized coords (0..1 for x,y; z is relative).
    """
    if ref is None or len(ref) != 21 or len(curr) != 21:
        return False
    for (cx, cy, cz), (rx, ry, rz) in zip(curr, ref):
        if abs(cx - rx) > tol: return False
        if abs(cy - ry) > tol: return False
        if abs(cz - rz) > tol: return False
    return True

def draw_label(img, text, org=(50, 100), scale=2.0, color=(0, 0, 255), thick=3):
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv.LINE_AA)

def print_hand_dump(idx, handed_label, handed_score, lms, img_shape):
    h, w = img_shape[:2]
    print(f"Hand {idx}: {handed_label} (score={handed_score:.2f})")
    for i, lm in enumerate(lms):
        x_px = int(lm.x * w)
        y_px = int(lm.y * h)
        print(f"  Lm {i}: norm=({lm.x:.3f},{lm.y:.3f},{lm.z:.3f}) px=({x_px},{y_px})")

def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    input_path = args.input
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    tolerance = args.tolerance

    # Setup capture
    is_image = False
    cap = None
    if input_path:
        if not os.path.exists(input_path):
            print(f"Input path not found: {input_path}")
            return
        if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            is_image = True
            image_input = cv.imread(input_path)
            if image_input is None:
                print(f"Failed to read image: {input_path}")
                return
            use_static_image_mode = True
        else:
            cap = cv.VideoCapture(input_path)
            cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    else:
        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # State
    ref_pose = list(DEFAULT_REF)  # list of 21 (x,y,z)
    prev_time = 0.0
    show_help = True

    with mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as hands:

        while True:
            # Acquire frame
            if not is_image:
                ret, image = cap.read()
                if not ret:
                    break
                image = cv.flip(image, 1)  # mirror for UX
            else:
                image = image_input.copy()
                image = cv.flip(image, 1)

            debug_image = copy.deepcopy(image)

            # Run MediaPipe
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            stop_detected = False

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Handedness
                    handedness_label = "Unknown"
                    handedness_score = 0.0
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        cls = results.multi_handedness[idx].classification[0]
                        handedness_label = cls.label
                        handedness_score = cls.score

                    # Check STOP (absolute pose match)
                    curr_norms = landmarks_to_list(hand_landmarks.landmark)
                    if all_within_tolerance(curr_norms, ref_pose, tol=tolerance):
                        stop_detected = True

                    # Optional: show small label per hand
                    h, w = image.shape[:2]
                    wrist = hand_landmarks.landmark[0]
                    p = (int(wrist.x * w), int(wrist.y * h))
                    cv.putText(debug_image, f"{handedness_label} {handedness_score:.2f}",
                               (p[0]+10, max(30, p[1]-10)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv.LINE_AA)

                if stop_detected:
                    draw_label(debug_image, "STOP \u270B", (50, 100), scale=2.0, color=(0,0,255), thick=3)

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0.0
            prev_time = curr_time
            cv.putText(debug_image, f'FPS: {int(fps)}', (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

            # Help overlay
            if show_help:
                help_lines = [
                    "Keys: ESC=quit, S=dump hand coords, R=set current hand as REF, H=toggle help",
                    f"Tolerance: {int(tolerance*100)}% per-axis on normalized x,y,z",
                ]
                y0 = 60
                for i, line in enumerate(help_lines):
                    cv.putText(debug_image, line, (10, y0 + 24*i),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv.LINE_AA)

            cv.imshow('Hand Gesture Recognition (STOP rule)', debug_image)

            key = cv.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == ord('h') or key == ord('H'):
                show_help = not show_help
            elif key == ord('s') or key == ord('S'):
                if results.multi_hand_landmarks:
                    # dump the FIRST detected hand
                    hand0 = results.multi_hand_landmarks[0]
                    handed_label = "Unknown"
                    handed_score = 0.0
                    if results.multi_handedness:
                        cls = results.multi_handedness[0].classification[0]
                        handed_label = cls.label
                        handed_score = cls.score
                    print_hand_dump(0, handed_label, handed_score, hand0.landmark, image.shape)
                else:
                    print("No hands to dump.")
            elif key == ord('r') or key == ord('R'):
                if results.multi_hand_landmarks:
                    # set reference from FIRST detected hand
                    hand0 = results.multi_hand_landmarks[0]
                    ref_pose = landmarks_to_list(hand0.landmark)
                    print("Reference pose updated from current hand (normalized x,y,z).")
                else:
                    print("No hands to capture as reference.")

            if is_image:
                # For single image input, wait for a key then exit
                if key != 255:  # any key pressed
                    break

    if cap is not None:
        cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
