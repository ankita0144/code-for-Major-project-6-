import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Helper function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def analyze_bicep_curl(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return {"exercise": "bicep_curl", "form": "no_pose_detected", "reps": 0}

    # Extract landmarks
    landmarks = results.pose_landmarks.landmark

    # Right arm points (you can use left if preferred)
    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    # Calculate angle
    angle = calculate_angle(shoulder, elbow, wrist)

    # Determine form
    form = "bad"
    if angle > 160:
        form = "down"
    if angle < 40:
        form = "up"

    # You can add logic here to count reps using global/static memory or session tracking

    return {
        "exercise": "bicep_curl",
        "angle": angle,
        "form": form,
        "reps": 0  # Actual counting would need stateful tracking (frontend or DB)
    }
