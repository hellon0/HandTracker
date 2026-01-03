import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

model_path = 'hand_landmarker.task'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)


detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17) # Palm base
])

def draw_landmarks_on_image(rgb_image, detection_result):
    handedness_list = detection_result.handedness
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = rgb_image.copy()
    height, width, _ = rgb_image.shape

    dot_color = (255, 0, 0)

    if len(handedness_list) > 0 and handedness_list[0][0].index == 0:
        dot_color = (0, 0, 255)
    

    # Iterate through the detected hands
    for hand_landmarks in hand_landmarks_list:
        # Draw the connections (lines)
        for connection in HAND_CONNECTIONS:
            start_node = hand_landmarks[connection[0]]
            end_node = hand_landmarks[connection[1]]
            # Scale normalized coordinates to image pixels
            start_point = (int(start_node.x * width), int(start_node.y * height))
            end_point = (int(end_node.x * width), int(end_node.y * height))
            cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2) # Green lines

        # Draw the landmarks (circles)
        for landmark in hand_landmarks:
            # Scale normalized coordinates to image pixels
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(annotated_image, (x, y), 5, dot_color, -1) # Blue circles
        
        dot_color = (-dot_color[0] + 255, 0, -dot_color[2] + 255)

    return annotated_image
    

webcam = cv2.VideoCapture(0)
while webcam.isOpened():
    success, img = webcam.read()

    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    detection_result = detector.detect_for_video(mp_image, int(webcam.get(cv2.CAP_PROP_POS_MSEC)))

    annotated_img = draw_landmarks_on_image(img, detection_result)


    cv2.imshow('Camera', annotated_img)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break


webcam.release()
cv2.destroyAllWindows()

