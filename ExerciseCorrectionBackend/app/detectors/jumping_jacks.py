import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def analyze_jumping_jacks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return {"exercise": "jumping_jacks", "form": "no_pose_detected", "reps": 0}

    landmarks = results.pose_landmarks.landmark

    # Get relevant body parts
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANK_
