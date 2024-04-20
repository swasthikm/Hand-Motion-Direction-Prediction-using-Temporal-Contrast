import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Object Motion Detection")

# Create a Label for displaying images
label = tk.Label(root)
label.pack()
direction_label = tk.Label(root, text="Direction: ")
direction_label.pack()
hand_detect_label = tk.Label(root, text="Hand: ")
hand_detect_label.pack()
# Open the webcam
cap = cv2.VideoCapture(0)
PIX_OFF_THRESH = -10
PIX_ON_THRESH = 10
imgSize = 300  # Define the size for resized images
offset = 20   # Offset for cropping the hand
detector = HandDetector(maxHands=1)

# Initialize direction and frame count variables
up_px_total = 0
down_px_total = 0
right_px_total = 0
left_px_total = 0
direction = "Initializing.... "
detect = "Searching..."
frame_count = 0
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
hand_detect = 0
while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_diff = frame_gray.astype(np.int16) - prev_frame_gray.astype(np.int16)
    thresholded_frame = np.where((frame_diff < PIX_OFF_THRESH), -1, 0)
    thresholded_frame = np.where(frame_diff > PIX_ON_THRESH, 1, thresholded_frame)
    output_frame = 128 + thresholded_frame.astype(np.uint8) * 127

    hands, _ = detector.findHands(frame, draw=False)

    if hands:
        hand = hands[0]
        hand_detect = hand_detect + 1
        x, y, w, h = hand['bbox']
        imgCrop = output_frame[y-offset:y + h + offset, x-offset:x + w + offset]

        if imgCrop.size == 0:
            detect = " Not Present in Frame"
            direction = " Fetching Direction ... "
            continue
        detect = " Detected "
        # Accumulate movement counts
        up_px_total += np.sum(imgCrop[:imgCrop.shape[0]//2, :] == 1)
        down_px_total += np.sum(imgCrop[imgCrop.shape[0]//2:, :] == 1)
        left_px_total += np.sum(imgCrop[:, imgCrop.shape[1]//2:] == 1)
        right_px_total += np.sum(imgCrop[:, :imgCrop.shape[1]//2] == 1)
        frame_count += 1

        # Make a prediction every 5 frames
        if frame_count == 5:
            max_count = max(up_px_total, down_px_total, left_px_total, right_px_total)
            if all(count < 5000 for count in [up_px_total, down_px_total, right_px_total, left_px_total]):
                direction = "No movement"
            elif up_px_total == max_count:
                direction = "Up"
            elif down_px_total == max_count:
                direction = "Down"
            elif left_px_total == max_count:
                direction = "Left"
            elif right_px_total == max_count:
                direction = "Right"
           # print("up:",up_px_total,"down:",down_px_total,"right:",right_px_total, "left:",left_px_total,"direction:", direction)
            # Reset the counters and frame count
            up_px_total = 0
            down_px_total = 0
            right_px_total = 0
            left_px_total = 0
            frame_count = 0
        cv2.rectangle(output_frame,(x-offset,y-offset),(x + w + offset, y+h + offset),(0,255,0),4)
    else:
        if(hand_detect != 0):
            direction = " Fetching Direction ... "
            detect = " Not Present in Frame"
    display_image = Image.fromarray(output_frame)
    photo = ImageTk.PhotoImage(image=display_image)
    label.config(image=photo)
    label.image = photo
    direction_label.config(text="Direction: " + direction)
    hand_detect_label.config(text = "Hand: " + detect)

    root.update_idletasks()
    root.update()

    prev_frame_gray = frame_gray.copy()
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
