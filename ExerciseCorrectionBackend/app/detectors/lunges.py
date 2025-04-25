import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    a = np.array(a)  # first point
    b = np.array(b)  # middle point (joint)
    c = np.array(c)  # end point

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def analyze_lunges(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return {"exercise": "lunges", "form": "no_pose_detected", "reps": 0}

    landmarks = results.pose_landmarks.landmark

    # Choose RIGHT leg for example
    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    # Calculate angle at knee
    angle = calculate_angle(hip, knee, ankle)

    # Determine form based on angle
    form = "bad"
    if 80 <= angle <= 100:
        form = "good"
    elif angle < 80:
        form = "too_deep"
    elif angle > 100:
        form = "not_deep_enough"

    return {
        "exercise": "lunges",
        "angle": angle,
        "form": form,
        "reps": 0
    }
