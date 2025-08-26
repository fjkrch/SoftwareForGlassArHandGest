import cv2
cap = cv2.VideoCapture(0)

ok, frame = cap.read()
if not ok:
    print("Webcam not accessible")
else:
    print("Webcam is working")
cap.release()
cv2.destroyAllWindows()