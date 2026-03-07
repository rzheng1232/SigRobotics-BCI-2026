import cv2

# Find available cameras
print("Looking for cameras...")
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()

# Open both cameras
cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

print("Press 'q' to quit")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1:
        cv2.imshow("Camera 1", frame1)
    if ret2:
        cv2.imshow("Camera 2", frame2)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
print("Cameras released, program ended.")